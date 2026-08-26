-- Fixture Hermes state store.
--
-- This is checked in as SQL rather than as a .db binary on purpose: you can
-- read it, review it in a diff, and see exactly what the demo is claiming.
-- `make fixture` compiles it to fixtures/state.fixture.db.
--
-- It reproduces the shape of a real pair of runs:
--
--   sess_cold_0001   the first time the agent did the task. It searched the
--                    web, read three files, ran five shell commands, hit a
--                    dead end, recovered, and then wrote a skill.
--   sess_skill_0002  the same task in a fresh session with that skill on disk.
--                    It loaded the skill and ran three commands.
--
-- The token numbers are deliberately confounded: the second run also got
-- prompt cache reads, which is exactly the situation where a naive total-token
-- delta overstates the skill. The report is expected to warn about it.
--
-- Columns follow the documented Hermes schema (see docs/MEASUREMENT.md). Only
-- the subset this project reads is populated.

PRAGMA foreign_keys = ON;

CREATE TABLE schema_version (
    version INTEGER NOT NULL
);
INSERT INTO schema_version (version) VALUES (22);

CREATE TABLE sessions (
    id                  TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    user_id             TEXT,
    model               TEXT,
    model_config        TEXT,
    system_prompt       TEXT,
    parent_session_id   TEXT,
    started_at          REAL NOT NULL,
    ended_at            REAL,
    end_reason          TEXT,
    message_count       INTEGER DEFAULT 0,
    tool_call_count     INTEGER DEFAULT 0,
    api_call_count      INTEGER DEFAULT 0,
    input_tokens        INTEGER DEFAULT 0,
    output_tokens       INTEGER DEFAULT 0,
    cache_read_tokens   INTEGER DEFAULT 0,
    cache_write_tokens  INTEGER DEFAULT 0,
    reasoning_tokens    INTEGER DEFAULT 0,
    billing_provider    TEXT,
    estimated_cost_usd  REAL,
    title               TEXT
);

CREATE TABLE messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    role          TEXT NOT NULL,
    content       TEXT,
    tool_call_id  TEXT,
    tool_calls    TEXT,
    tool_name     TEXT,
    timestamp     REAL NOT NULL,
    token_count   INTEGER,
    finish_reason TEXT
);

-- ---------------------------------------------------------------------------
-- Run 1: cold. No skill on disk. 11 tool calls, no cache reads.
-- ---------------------------------------------------------------------------
INSERT INTO sessions VALUES (
    'sess_cold_0001', 'cli', 'local', 'anthropic/claude-opus-4', NULL, NULL, NULL,
    1756100000.0, 1756100742.0, 'completed',
    26,          -- message_count
    11,          -- tool_call_count
    12,          -- api_call_count
    48200,       -- input_tokens
    3100,        -- output_tokens
    0,           -- cache_read_tokens
    9800,        -- cache_write_tokens
    0,
    'anthropic', 0.4127,
    'Cut an RC for the api service'
);

-- Run 2: same task, fresh session, skill present. 4 tool calls, cache warm.
INSERT INTO sessions VALUES (
    'sess_skill_0002', 'cli', 'local', 'anthropic/claude-opus-4', NULL, NULL, NULL,
    1756180000.0, 1756180131.0, 'completed',
    11,          -- message_count
    4,           -- tool_call_count
    5,           -- api_call_count
    41000,       -- input_tokens
    1400,        -- output_tokens
    31000,       -- cache_read_tokens  <- most of the input was cached
    0,           -- cache_write_tokens
    0,
    'anthropic', 0.0611,
    'Cut an RC for the api service'
);

-- A third session that is NOT part of the comparison, so `sessions` has
-- something to list and you can see the picker working.
INSERT INTO sessions VALUES (
    'sess_gateway_0003', 'telegram', 'u_1729', 'anthropic/claude-opus-4',
    NULL, NULL, NULL,
    1756190000.0, 1756190090.0, 'completed',
    6, 2, 3, 12400, 700, 9100, 0, 0,
    'anthropic', 0.0208,
    'Check staging health'
);

-- ---------------------------------------------------------------------------
-- Tool rows. role='tool' is what counted_tool_calls() recounts, so these have
-- to match tool_call_count above or the report raises a discrepancy warning.
-- ---------------------------------------------------------------------------

-- Cold run: 2 web_search + 3 read_file + 5 terminal + 1 skill_manage = 11
INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES
    ('sess_cold_0001', 'user',      'Cut an RC for the api service', NULL, 1756100001.0),
    ('sess_cold_0001', 'assistant', 'Looking up the release process.', NULL, 1756100005.0),
    ('sess_cold_0001', 'tool', 'results', 'web_search',   1756100010.0),
    ('sess_cold_0001', 'tool', 'results', 'web_search',   1756100024.0),
    ('sess_cold_0001', 'tool', 'file',    'read_file',    1756100060.0),
    ('sess_cold_0001', 'tool', 'file',    'read_file',    1756100091.0),
    ('sess_cold_0001', 'tool', 'file',    'read_file',    1756100120.0),
    ('sess_cold_0001', 'tool', 'ok',      'terminal',     1756100180.0),
    ('sess_cold_0001', 'tool', 'empty changelog', 'terminal', 1756100240.0),
    ('sess_cold_0001', 'assistant', 'That produced an empty changelog. The base tag was wrong.', NULL, 1756100250.0),
    ('sess_cold_0001', 'tool', 'ok',      'terminal',     1756100300.0),
    ('sess_cold_0001', 'tool', 'ok',      'terminal',     1756100420.0),
    ('sess_cold_0001', 'tool', 'ok',      'terminal',     1756100600.0),
    ('sess_cold_0001', 'tool', 'created', 'skill_manage', 1756100700.0),
    ('sess_cold_0001', 'assistant', 'Tagged v1.8.0-rc3 and saved a skill for next time.', NULL, 1756100740.0);

-- Skill run: 1 skill_view + 3 terminal = 4
INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES
    ('sess_skill_0002', 'user',      'Cut an RC for the api service', NULL, 1756180001.0),
    ('sess_skill_0002', 'tool', 'skill body', 'skill_view', 1756180008.0),
    ('sess_skill_0002', 'tool', 'ok', 'terminal', 1756180030.0),
    ('sess_skill_0002', 'tool', 'ok', 'terminal', 1756180070.0),
    ('sess_skill_0002', 'tool', 'ok', 'terminal', 1756180110.0),
    ('sess_skill_0002', 'assistant', 'Tagged v1.8.0-rc4. Changelog base tag taken from the skill.', NULL, 1756180129.0);

INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES
    ('sess_gateway_0003', 'user', 'is staging up', NULL, 1756190001.0),
    ('sess_gateway_0003', 'tool', 'ok', 'terminal', 1756190020.0),
    ('sess_gateway_0003', 'tool', 'ok', 'terminal', 1756190050.0),
    ('sess_gateway_0003', 'assistant', 'Staging is healthy.', NULL, 1756190085.0);
