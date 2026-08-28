# Northwind Freight, systems overview

*Internal wiki page. Owner: IT Operations. Last reviewed: see bottom.*

This is Northwind Freight's environment. You did not build it and you cannot
change it. Requests to "just fix the API" have been raised eleven times since
2021 and the platform replacement is still on the roadmap for next financial
year, as it has been for four financial years.

## Dispatch API

The dispatch platform exposes a small HTTP API. It went live in 2019. The
engineer who wrote it left the business shortly afterwards and there is no
formal spec, so this page is what we have.

Base URL in the sandbox: `http://localhost:8000`

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | Liveness. Always available. |
| `/api/v1/shipments` | GET | List of shipments. Paginated, `offset` and `limit`. |
| `/api/v1/shipment/{id}` | GET | Single shipment detail. |
| `/api/v1/drivers` | GET | Driver roster by depot. |
| `/api/v1/shipment/{id}/reschedule` | POST | Books a redelivery. Requires `X-Api-Key`. |
| `/api/v1/reschedules` | GET | Reschedules created in this session. |

Sandbox API key: `northwind-dev-key`. Production keys are issued by IT
Operations and are per integration, never shared.

All endpoints return JSON.

There is a rate limit. Nobody currently in the team knows what the threshold
is. If you start seeing `429`, back off and read the `Retry-After` header.

## Operational exports

`data/shipments_2024.csv` and `data/shipments_2025.csv` are dumps from the
warehouse reporting job. They are produced weekly and land on the shared
drive. Both files are comma separated with a header row.

Known issues, as reported by the analytics team:

- The 2025 file came out of the platform migration and does not match the
  2024 layout exactly. A column was renamed and a column was added.
- Date formatting is not consistent through the 2025 file. The migration
  changed the format partway through the year.
- Some rows contain characters that Excel renders as question marks. This is
  believed to be a Windows locale setting on the reporting server.
- Empty values are not represented consistently. Treat anything that looks
  empty as empty.

Analytics load these files in Excel and have not reported row loss, so the
files are believed to be structurally sound.

## Policy documents

`data/policies/` holds the five customer facing policy documents that the
service desk works from. They are maintained by different owners and are
versioned independently, which means the effective dates do not line up.

## Support tickets

`tickets.jsonl` is an export from the service desk tool, one JSON object per
line. Free text, unstructured, and written by whoever was on shift. It is the
best record we have of why deliveries actually fail.

---

*Reviewed annually. Last review date is recorded in the change log, which
moved during the intranet migration and has not been relocated. Assume parts
of this page are out of date. Customer documentation usually is, and where
this page disagrees with the system, the system is right.*
