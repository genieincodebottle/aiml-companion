// The journeys a first-time user takes, end to end through the real UI.
import { expect, test } from '@playwright/test';

async function actAs(page, name) {
  await page.getByLabel('Acting as').selectOption(name);
}

async function investigate(page, alertId) {
  const card = page.locator('.alert-card', { hasText: alertId });
  await card.getByRole('button', { name: 'Investigate' }).click();
}

async function investigateWithFault(page, alertId) {
  const card = page.locator('.alert-card', { hasText: alertId });
  await card.getByRole('button', { name: '+ fault' }).click();
}

const gate = (page) => page.getByText('Reviewer sign-off required');

// Any console error or page error fails the test: leaked props, failed requests and
// render loops get caught in CI instead of by a user.
let problems = [];
test.beforeEach(async ({ page }) => {
  problems = [];
  page.on('pageerror', (e) => problems.push(`page error: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console error: ${m.text()}`);
  });
  await page.goto('/');
  await actAs(page, 'ana.analyst');
});
test.afterEach(() => {
  expect(problems, problems.join(' | ')).toEqual([]);
});

test('offline mode is labelled', async ({ page }) => {
  await expect(page.getByText('Offline mode.')).toBeVisible();
  await expect(page.getByText('offline mode', { exact: true })).toBeVisible();
});

test('the analyst cannot approve their own case, a reviewer can, and is recorded', async ({
  page,
}) => {
  await investigate(page, 'ALRT-001');
  await expect(gate(page)).toBeVisible();
  await expect(page.getByText('You opened this case')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Approve', exact: true })).toBeDisabled();

  await actAs(page, 'rex.reviewer');
  await expect(page.getByRole('button', { name: 'Approve', exact: true })).toBeEnabled();
  await page.getByPlaceholder(/Why you decided/).fill('sanctions hit confirmed');
  await page.getByRole('button', { name: 'Approve', exact: true }).click();
  await expect(page.locator('.inv-stats').getByText('Resolved')).toBeVisible();
  await expect(page.getByText(/approved by rex\.reviewer/)).toBeVisible();
  await expect(page.getByText('sanctions hit confirmed')).toBeVisible();
});

test('rejecting escalates and files nothing', async ({ page }) => {
  await investigate(page, 'ALRT-002');
  await expect(gate(page)).toBeVisible();
  await actAs(page, 'rex.reviewer');
  await page.getByRole('button', { name: 'Reject & escalate' }).click();
  await expect(page.locator('.inv-stats').getByText('Escalated')).toBeVisible();
  await expect(page.getByText(/rejected by rex\.reviewer/)).toBeVisible();
});

test('a commit fault compensates instead of half-filing', async ({ page }) => {
  await investigateWithFault(page, 'ALRT-001');
  await expect(gate(page)).toBeVisible();
  await actAs(page, 'rex.reviewer');
  await page.getByRole('button', { name: 'Approve', exact: true }).click();
  await expect(page.locator('.inv-stats').getByText('Compensated')).toBeVisible();
  await expect(page.locator('.timeline').getByText('saga compensated')).toBeVisible();
});

test('the planted instruction is caught on the timeline', async ({ page }) => {
  await investigate(page, 'ALRT-004');
  await expect(gate(page)).toBeVisible();
  await expect(page.locator('.timeline').getByText('injection blocked')).toBeVisible();
});

test('a parked case appears in recent cases and can be reopened', async ({ page }) => {
  await investigate(page, 'ALRT-005');
  await expect(gate(page)).toBeVisible();
  await page.reload();
  await page.locator('.case-row', { hasText: 'ALRT-005' }).first().click();
  await expect(gate(page)).toBeVisible();
});
