// Renders every page against every recorded trace, server side.
//
//    npm run smoke
//
// A vite build proves the code parses. It does not prove a page survives contact
// with a real trace, and the traces here are the awkward ones: a killed branch,
// a timeout, a saga with five extra spans, a clean run with nothing broken.

import { readFileSync, readdirSync } from 'node:fs'
import React from 'react'
import { renderToString } from 'react-dom/server'
import Overview from './pages/Overview.jsx'
import Scenarios from './pages/Scenarios.jsx'
import TracePage from './pages/TracePage.jsx'
import CostPage from './pages/CostPage.jsx'
import BreakIt from './pages/BreakIt.jsx'

const dir = '../backend/replay/'
const load = (f) => JSON.parse(readFileSync(dir + f, 'utf-8'))
const files = readdirSync(dir).filter((f) => f.endsWith('.json')).sort()
const go = () => {}
let failed = 0

const config = {
  prices_last_checked: '2026-08-04',
  mode: 'replay',
  failure_catalog: [
    { key: 'kill_shipping_agent', label: 'Kill it', stage: 4, type: 'bool', teaches: 't', watch: 'w' },
    { key: 'slow_tool_seconds', label: 'Slow it', stage: 4, type: 'seconds', default_on: 12, teaches: 't', watch: 'w' },
  ],
}

function check(name, el, expect) {
  try {
    const html = renderToString(el)
    const ok = expect ? expect(html) : html.length > 0
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}`)
    if (!ok) failed++
    return html
  } catch (e) {
    console.log(`FAIL ${name}: ${e.message}`)
    failed++
    return ''
  }
}

// Pages that do not need a result.
check('overview', <Overview go={go} mode="replay" />, (h) => h.includes('Start the tour'))
check('scenarios (empty)', <Scenarios replays={[]} onPick={go} activeId={null} />)
check('trace (no result)', <TracePage result={null} go={go} />, (h) => h.includes('Nothing loaded yet'))
check('cost (no result)', <CostPage result={null} config={config} go={go} />)
check('break it', <BreakIt catalog={config.failure_catalog} failures={{}} setFailures={go} question="q" setQuestion={go} onRun={go} busy={false} mode="replay" />, (h) => h.includes('Run it and show me the trace'))

// Every recorded trace through the trace and cost pages.
const replays = files.map((f) => {
  const d = load(f)
  return { id: d.id, title: d.title, teaches: d.teaches, failures: d.failures, question: d.question }
})
check('scenarios (full)', <Scenarios replays={replays} onPick={go} activeId={replays[0]?.id} />, (h) => h.includes('View trace'))

for (const f of files) {
  const result = load(f)
  const rows = check(`trace: ${result.id}`, <TracePage result={result} go={go} />, (h) => h.includes('class="row'))
  // The header renders as `head row`, so it never matches this pattern and
  // needs no subtracting. One match per span is exactly right.
  const spanRows = (rows.match(/class="row[ "]/g) || []).length
  const expected = result.trace.spans.length
  const okRows = spanRows === expected
  console.log(`     ${okRows ? 'ok  ' : 'FAIL'} rows ${spanRows}/${expected}`)
  if (!okRows) failed++

  if (result.id === '05-green-and-wrong') {
    const green = rows.includes('There\u0020is no status code for wrong') || rows.includes('no status code for wrong')
    console.log(`     ${green ? 'ok  ' : 'FAIL'} green-and-wrong banner present`)
    if (!green) failed++
  }
  check(`cost:  ${result.id}`, <CostPage result={result} config={config} go={go} />, (h) => h.includes('this question'))
}

console.log(failed === 0 ? '\nall checks passed' : `\n${failed} CHECK(S) FAILED`)
process.exit(failed === 0 ? 0 : 1)
