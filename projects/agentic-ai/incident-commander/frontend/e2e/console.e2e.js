// The journeys a first-time user takes, end to end through the real UI.
import { expect, test } from '@playwright/test';

async function trigger(page, scenario) {
  await page.goto('/console');
  await page.getByLabel('Fault scenario').click();
  await page.getByRole('option', { name: scenario }).click();
  await page.getByRole('button', { name: 'Trigger Incident' }).click();
}

const gate = (page) => page.getByText('Your decision is needed');

// Any console error or React warning fails the test. That is how leaked props,
// failed requests and render loops get caught in CI instead of by a user.
let problems = [];
test.beforeEach(async ({ page }) => {
  problems = [];
  page.on('pageerror', (e) => problems.push(`page error: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console error: ${m.text()}`);
  });
});
test.afterEach(() => {
  expect(problems, problems.join(' | ')).toEqual([]);
});

test('offline mode is labelled on every page', async ({ page }) => {
  for (const path of ['/home', '/console', '/flow', '/history']) {
    await page.goto(path);
    await expect(page.getByText('Offline mode.')).toBeVisible();
  }
});

test('a risky fix waits for a human, and the postmortem names the approver', async ({ page }) => {
  await trigger(page, /Bad Deploy/);
  await expect(gate(page)).toBeVisible();
  // Nothing has run yet: the rollback is high risk.
  await page.getByRole('button', { name: 'Approve', exact: true }).click();
  await expect(page.getByText('Resolved', { exact: true })).toBeVisible();
  await expect(page.getByText(/rollback_deploy .* approved by anonymous/)).toBeVisible();
});

test('rejecting the fix escalates instead of acting', async ({ page }) => {
  await trigger(page, /Bad Deploy/);
  await expect(gate(page)).toBeVisible();
  await page.getByRole('button', { name: 'Reject', exact: true }).click();
  await expect(page.getByText('Escalated to a human')).toBeVisible();
  await expect(page.getByText(/rejected by anonymous/)).toBeVisible();
});

test('cancel at the gate stops the run', async ({ page }) => {
  await trigger(page, /Bad Deploy/);
  await expect(gate(page)).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByText('Cancelled by operator')).toBeVisible();
});

test('a finished incident appears in History', async ({ page }) => {
  await trigger(page, /False Alarm/);
  // Even "do nothing" passes the human gate unless auto-approve is on.
  await expect(gate(page)).toBeVisible();
  await page.getByRole('button', { name: 'Approve', exact: true }).click();
  await expect(page.getByText('Resolved', { exact: true })).toBeVisible();
  await page.goto('/history');
  await expect(page.getByText('false_alarm').first()).toBeVisible();
});
