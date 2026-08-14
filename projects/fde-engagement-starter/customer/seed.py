"""Deterministic generator for the Northwind Freight customer environment.

Run:  python customer/seed.py

Regenerates, byte for byte identically on every run:

    customer/data/shipments_2024.csv
    customer/data/shipments_2025.csv
    customer/data/policies/*.md
    customer/tickets.jsonl

Everything is seeded from random.seed(42). Two runs produce identical bytes,
so results are comparable between learners and tests can pin exact rows.

The exports are written as raw bytes on purpose. They are not a clean
DataFrame.to_csv dump, because a real operational export from a 2019 system
is not one either.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
BOM_UTF8 = b"\xef\xbb\xbf"  # utf-8 encoding of U+FEFF
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POLICIES = DATA / "policies"

CITIES = [
    "Rotterdam", "Antwerp", "Hamburg", "Bremen", "Lyon", "Marseille",
    "Bilbao", "Porto", "Turin", "Gdansk", "Utrecht", "Leeds",
    "Manchester", "Dublin", "Aarhus", "Malmo",
]

# Cities whose names carry accents. These rows get written as latin-1 bytes
# inside an otherwise utf-8 file.
LATIN1_CITIES = ["Malaga", "Nimes", "Koln", "Orleans", "Munster"]
LATIN1_ACCENTED = {
    "Malaga": "Málaga",
    "Nimes": "Nîmes",
    "Koln": "Köln",
    "Orleans": "Orléans",
    "Munster": "Münster",
}

STATUSES = ["delivered", "delivered", "delivered", "failed", "in_transit", "returned"]
FAILURE_REASONS = [
    "address_incorrect",
    "recipient_absent",
    "damaged_in_transit",
    "customs_hold",
    "site_access_denied",
]
SERVICE_TIERS = ["standard", "express", "economy", "express_am"]
NULL_TOKENS = ["", "NULL", "N/A", "-", "nan"]

STREETS = [
    "Main St", "Harbour Rd", "Kingsway", "Station Rd", "Mill Lane",
    "Dock Approach", "Bridge St", "Elm Grove",
]


def _addr(rng: random.Random, embed_comma: bool) -> str:
    """Build a destination address. If embed_comma, the field must be quoted."""
    num = rng.randint(1, 240)
    street = rng.choice(STREETS)
    if embed_comma:
        unit = rng.randint(1, 30)
        return '"%d %s, Unit %d"' % (num, street, unit)
    return "%d %s" % (num, street)


def _maybe_null(rng: random.Random, value: str, p: float) -> str:
    """Replace value with one of five inconsistent null spellings."""
    if rng.random() < p:
        return rng.choice(NULL_TOKENS)
    return value


def _date_2024(rng: random.Random) -> str:
    return "2024-%02d-%02d" % (rng.randint(1, 12), rng.randint(1, 28))


def _date_2025(rng: random.Random, row_index: int) -> str:
    """ISO for the first half of the file, then DD/MM/YYYY for the rest.

    The system that produced this export was migrated mid year and nobody
    reprocessed the earlier rows.
    """
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    if row_index < 200:
        return "2025-%02d-%02d" % (month, day)
    return "%02d/%02d/2025" % (day, month)


def _build_2024(rng: random.Random) -> bytes:
    header = (
        "shipment_id,customer_id,origin_city,dest_city,dest_address,"
        "ship_date,delivery_date,status,failure_reason,weight_kg"
    )
    lines = [header]
    for i in range(400):
        sid = "SH24%05d" % (10000 + i)
        cust = 4000 + rng.randint(0, 1500)
        origin = rng.choice(CITIES)
        dest = rng.choice(CITIES)
        addr = _addr(rng, embed_comma=(i % 61 == 3))
        ship = _date_2024(rng)
        deliv = _maybe_null(rng, _date_2024(rng), 0.10)
        status = rng.choice(STATUSES)
        reason = rng.choice(FAILURE_REASONS) if status == "failed" else rng.choice(NULL_TOKENS)
        weight = _maybe_null(rng, "%.2f" % (rng.random() * 90 + 0.5), 0.05)
        lines.append(
            ",".join([sid, str(cust), origin, dest, addr, ship, deliv, status, reason, weight])
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_2025(rng: random.Random) -> bytes:
    """2025 export. Same idea as 2024, with every defect the migration added.

    Written as a byte stream because parts of it are not utf-8.
    """
    header = (
        "shipment_id,cust_id,origin_city,dest_city,dest_address,"
        "ship_date,delivery_date,status,failure_reason,weight_kg,service_tier"
    )
    buf = io.BytesIO()
    buf.write(BOM_UTF8)  # byte order mark, first bytes of the file
    buf.write((header + "\n").encode("utf-8"))

    # Row indexes that get structural damage. Fixed, not random, so tests can pin them.
    extra_field_rows = {57, 158, 291}          # more fields than the header
    missing_field_rows = {74, 233}             # fewer fields than the header
    embedded_comma_rows = {12, 40, 88, 141, 202, 260, 333}  # quoted field with a delimiter
    latin1_rows = {33, 96, 177, 249, 310, 371}  # written as latin-1 bytes

    for i in range(400):
        sid = "SH25%05d" % (10000 + i)
        # Zero padded string customer id. 2024 stored the same value as an int.
        cust = '"%07d"' % (4000 + rng.randint(0, 1500))
        origin = rng.choice(CITIES)
        dest = rng.choice(CITIES)
        addr = _addr(rng, embed_comma=(i in embedded_comma_rows))
        ship = _date_2025(rng, i)
        deliv = _maybe_null(rng, _date_2025(rng, i), 0.10)
        status = rng.choice(STATUSES)
        reason = rng.choice(FAILURE_REASONS) if status == "failed" else rng.choice(NULL_TOKENS)
        weight = _maybe_null(rng, "%.2f" % (rng.random() * 90 + 0.5), 0.05)
        tier = _maybe_null(rng, rng.choice(SERVICE_TIERS), 0.04)

        encoding = "utf-8"
        if i in latin1_rows:
            plain = LATIN1_CITIES[sorted(latin1_rows).index(i) % len(LATIN1_CITIES)]
            dest = LATIN1_ACCENTED[plain]
            encoding = "latin-1"

        fields = [sid, cust, origin, dest, addr, ship, deliv, status, reason, weight, tier]

        if i in extra_field_rows:
            # The upstream job appended two operational columns for a fortnight
            # and never told anyone. No header change.
            fields.append("depot_%d" % rng.randint(1, 9))
            fields.append("scan_%d" % rng.randint(100, 999))
        elif i in missing_field_rows:
            # Truncated write. The tail of the row is simply absent.
            fields = fields[:-2]

        buf.write((",".join(fields) + "\n").encode(encoding))

    return buf.getvalue()


# --------------------------------------------------------------------------
# Policy documents
# --------------------------------------------------------------------------

POLICY_DOCS = {
"delivery-sla.md": """# Northwind Freight Delivery Service Level Agreement

Document owner: Operations Standards Group
Version 4.2, effective January 2025
Supersedes version 4.1

## 1. Scope

This agreement covers all ground and multimodal freight movements booked
through the Northwind dispatch platform for customers on standard, express
and economy service tiers. Air charter and dedicated fleet arrangements are
governed by their own contracts and are out of scope.

## 2. Committed transit windows

Express shipments are committed for next business day delivery where both
origin and destination lie within the core European network. Standard
shipments are committed to a three business day window. Economy shipments
carry a five business day window and are explicitly not guaranteed during
declared peak periods.

Transit clocks start at the first depot scan, not at the point of booking.
Customers frequently misread this and open disputes on that basis, so
service desk staff should quote the depot scan timestamp when responding.

## 3. Failed delivery handling

Where a driver is unable to complete a delivery, the shipment is returned to
the local depot and a delivery exception is raised automatically. The
shipment is then placed on the standard retry cycle.

Northwind attempts a total of two redelivery attempts on consecutive
business days following the original exception. If the second attempt also
fails, the shipment moves to a hold state at the destination depot and the
consignee is contacted for collection or for a corrected address.

Shipments held for more than ten calendar days are returned to origin at
the shipper's cost.

## 4. Exclusions

The committed windows above are suspended where delivery is prevented by
customs intervention, adverse weather declared at national level, industrial
action, or an incorrect or incomplete address supplied by the shipper. Time
spent in a customs hold does not count toward the transit clock.

## 5. Service credits

Where a committed window is missed for reasons inside Northwind's control,
the customer may claim a service credit against the freight charge for the
affected consignment. Credits are applied to the following month's invoice
and are not paid in cash.
""",

"refund-policy.md": """# Refund and Service Credit Policy

Document owner: Commercial Finance
Version 2.7, effective March 2025

## Purpose

This policy sets out when Northwind Freight refunds freight charges, when it
issues a service credit instead, and who may approve each.

## Refund versus credit

A refund returns money to the payment instrument used at booking. A service
credit reduces a future invoice. Northwind's default remedy is a service
credit. A cash refund is issued only where the customer has no ongoing
account, or where the consignment was never collected.

## Eligibility

Freight charges are refundable in full where:

- The consignment was never collected and the booking was cancelled before
  the first depot scan.
- The consignment was lost in the network and a loss declaration has been
  issued by the claims team.
- Northwind cancelled the movement without a substitute service.

Freight charges are refundable in part, normally at fifty percent, where the
consignment was delivered outside its committed window for reasons inside
Northwind's control, and the customer did not receive a service credit for
the same consignment.

## Not refundable

Duties, taxes, brokerage fees and any surcharge levied by a third party are
never refundable by Northwind. Charges arising from an address correction
requested after collection are not refundable. Storage charges accrued while
a shipment sits on hold are not refundable regardless of the cause of the
hold.

## Claim window

Claims must be submitted within twenty eight calendar days of the delivery
date or, for undelivered consignments, within twenty eight days of the last
tracking event. Claims received after this window are rejected without
review. The service desk has no discretion to extend it.

## Approval

Credits up to five hundred euro are approved by the account manager. Above
that figure, approval passes to Commercial Finance. Cash refunds of any
value require Commercial Finance approval.
""",

"hazardous-goods.md": """# Hazardous Goods Acceptance Standard

Document owner: Safety and Compliance
Version 1.9, effective November 2024

## Applicability

This standard applies to any consignment containing goods classified as
dangerous under ADR. It applies to every Northwind depot, every subcontracted
carrier operating under a Northwind consignment note, and every customer
booking through the dispatch platform.

## Booking requirements

Dangerous goods may not be booked through the self service channel. The
customer must book through their account manager and must supply a completed
dangerous goods note before collection is scheduled. A booking that reaches a
depot without a dangerous goods note is refused at the dock and the customer
is charged an abortive collection fee.

## Classes accepted

Northwind accepts limited quantity shipments in classes 2, 3, 8 and 9 on the
standard network. Classes 1, 6.2 and 7 are never accepted. Class 4 and 5
material is accepted only on dedicated movements arranged in advance.

Lithium batteries shipped separately from equipment are treated as class 9
and require the relevant mark and handling label regardless of quantity.

## Segregation and handling

Depot staff must observe the segregation table published in the operations
handbook. Dangerous goods must not be left in a trailer overnight at a site
without a fire plan. Any leak, spill or damaged package is an immediate stop
work event.

## Incidents

A dangerous goods incident is reported to the duty compliance officer within
one hour, regardless of severity and regardless of the hour. The reporting
route is the compliance hotline, not the standard operations escalation path.
The shipment is quarantined until compliance releases it. No redelivery
attempt may be made on a quarantined shipment.

## Records

Dangerous goods notes are retained for five years. Drivers' training records
are retained for the duration of employment plus three years.
""",

"address-correction.md": """# Address Correction Procedure

Document owner: Service Desk Operations
Version 3.1, effective February 2025

## When this applies

Use this procedure when a consignment cannot be delivered because the
address supplied at booking is wrong, incomplete, or does not match a
deliverable location. It does not apply where the address is correct but the
recipient was absent, which is handled under the standard retry cycle.

## Detecting the case

Drivers raise an exception code at the point of failure. Codes ADDR1 through
ADDR4 indicate an address problem. ADDR1 is a house or unit number that does
not exist. ADDR2 is a postcode that does not match the street. ADDR3 is a
business that has moved or closed. ADDR4 is a site the driver could not
locate at all.

Note that drivers under time pressure sometimes record an access problem as
ADDR4. The service desk should read the free text comment before treating an
ADDR4 as an address fault.

## Obtaining a correction

Contact the shipper first, not the consignee. The shipper is the contracting
party and owns the address data. Only where the shipper cannot be reached
within one working day may the service desk contact the consignee directly.

A corrected address is recorded against the consignment in the dispatch
platform. The original address is retained. Corrections are never applied
retrospectively to the customer's address book, because Northwind does not
own that data.

## Charges

An address correction applied after collection attracts a correction fee per
consignment. The fee is charged to the shipper's account, not to the
consignee, even where the consignee supplied the correction.

## Redelivery after correction

Once a corrected address is recorded, the consignment re-enters the retry
cycle at the next available delivery run. A correction does not reset the
committed transit window, and no service credit arises from a delay caused
by a bad address.
""",

"escalation-matrix.md": """# Escalation Matrix

Document owner: Customer Operations
Version 5.0, effective April 2025

## How to use this document

Escalation is time based. Find the elapsed time since the triggering event,
then read across. Do not escalate on customer pressure alone. Do not skip a
tier, except for the safety cases listed at the end.

## Tier 1, service desk

All customer contact starts here. The service desk owns the case until it is
closed or escalated. Tier 1 may reschedule a delivery, record an address
correction, raise a claim, and issue a service credit up to five hundred
euro.

Tier 1 handles the full redelivery cycle. A shipment that fails on delivery
is retried on three consecutive business days before it is placed on hold at
the destination depot. If all three attempts fail, the case escalates to
tier 2 automatically and the consignee is contacted for collection.

## Tier 2, operations duty manager

Engaged after twenty four hours of an unresolved tier 1 case, or immediately
where the consignment is temperature controlled, high value, or already the
subject of a formal complaint. The duty manager may authorise a dedicated
vehicle, a depot to depot transfer, or an out of network courier.

## Tier 3, regional operations lead

Engaged after seventy two hours, or where more than five consignments for
one customer are affected by a single incident. The regional lead owns
communication with the account manager from this point.

## Tier 4, director on call

Engaged for network wide disruption, regulatory contact, media contact, or
any event with a credible route to the national press.

## Safety exceptions

A dangerous goods incident, a road traffic collision involving injury, or a
suspected security breach bypasses every tier above and goes directly to the
duty compliance officer on the compliance hotline. Tier 1 does not triage
these cases and must not attempt to resolve them.
""",
}


# --------------------------------------------------------------------------
# Support tickets
# --------------------------------------------------------------------------

TICKET_TEMPLATES = {
    "wrong_address": [
        "Driver says the unit number does not exist. It is {num} {street}, the depot has {num2} on the label. Please fix and redeliver.",
        "we booked this to the new warehouse and it went to the old one again. address on the account is out of date, i have told you three times now. can someone actually update it",
        "Consignment {sid} came back marked ADDR2. Postcode on the label does not match the street. I have attached the correct address, please rebook for the next available run and confirm the correction fee will not be charged as this came from your side.",
        "Address wrong. Fix it.",
        "The site moved in January. Everything sent to the old address is bouncing. This is the fourth one this month and each time we are told the address book is updated and each time it is not. I would like someone to call me rather than send another templated reply, because at this point the freight cost is less of a problem than the time we spend chasing it.",
    ],
    "missing_recipient": [
        "Nobody was in when the driver called. Can you leave it with a neighbour next time or leave it in the porch, we are happy to take the risk.",
        "second failed attempt, no card left either. our reception is staffed 9 to 5 and the driver came at 7pm which is not a delivery window anyone agreed to.",
        "Recipient absent again on {sid}. Please hold at depot, we will collect Friday.",
        "no answer at the door apparently. i was working from home all day. no knock, no card, nothing. i think the driver did not attempt it.",
        "Delivery attempted while the site was closed for the bank holiday. Please reschedule for Tuesday and note our closure dates on the account so this stops happening.",
    ],
    "damaged_goods": [
        "Two of the four cartons on {sid} arrived crushed. Contents unusable. Photos attached, please advise on the claim process.",
        "box arrived open and taped back up with your tape. items missing. this is not acceptable",
        "Pallet {sid} was delivered with visible water damage to the lower layer. The driver noted it on the POD. We have refused the affected cartons and accepted the rest. Please confirm the credit and arrange collection of the refused goods.",
        "damaged. want refund.",
        "The consignment arrived intact but the packaging shows it was dropped. We are accepting it under reservation. Logging this so there is a record if the equipment fails on commissioning next week, since the warranty conversation will be easier with a ticket number attached.",
    ],
    "customs_hold": [
        "Shipment {sid} has been sitting at customs for six days. Tracking has not moved. What is missing from the paperwork?",
        "customs are asking for a commercial invoice with the harmonised codes. we sent this at booking. can you resend it to them rather than back to us",
        "Held by customs. Duty invoice received but the value looks wrong, it is showing the insured value not the declared value. Please correct before we pay.",
        "Why is this in customs at all, it is an intra EU movement. Someone has classified it incorrectly.",
        "We have been told the hold is because the consignee VAT number failed validation. The number is correct and has been in use for eleven years. I suspect the broker has transposed two digits. Please check the entry and come back to me today, the goods are time sensitive and we are into storage charges tomorrow.",
    ],
    "site_access": [
        "Driver could not get onto the site, the barrier needs a code. Code is on the booking notes. Please make sure it reaches the driver.",
        "our loading bay is only accessible for vehicles under 7.5t. you sent an artic again. it cannot physically turn into the yard.",
        "Access denied at {sid}, security would not admit the driver without a pre booked slot. We book slots through the portal, the reference is on the consignment note.",
        "no access. gate locked.",
        "The delivery point is a construction site and requires the driver to hold a valid site card, which was stated at booking and appears on the consignment note. The driver who attended today did not have one and was correctly turned away. Please send a carded driver and do not charge us for the failed attempt.",
    ],
}

MISFILED_TICKETS = [
    "Can you send me a copy of last month's invoice, I cannot find it in the portal.",
    "please remove me from your marketing emails",
    "I think I have been charged twice for the same booking in June. Reference on the statement is NF-88213.",
    "How do I add a second user to our account? The person who set it up has left.",
    "Testing whether this address works, please ignore.",
]

CHANNELS = ["email", "email", "email", "phone", "webform", "chat"]

RESOLUTIONS = {
    "wrong_address": ["address_corrected_redelivered", "returned_to_shipper", "pending_shipper_contact"],
    "missing_recipient": ["redelivered", "held_for_collection", "returned_to_shipper"],
    "damaged_goods": ["claim_raised", "credit_issued", "refused_and_collected"],
    "customs_hold": ["documents_resubmitted", "cleared", "pending_broker"],
    "site_access": ["redelivered_with_slot", "vehicle_swap_arranged", "pending_customer_info"],
    "misc": ["routed_to_billing", "routed_to_accounts", "closed_no_action"],
}


def _build_tickets(rng: random.Random) -> str:
    modes = sorted(TICKET_TEMPLATES.keys())
    lines = []
    for i in range(120):
        tid = "NF-T%05d" % (20250 + i)
        created = "2025-%02d-%02dT%02d:%02d:00Z" % (
            rng.randint(1, 12), rng.randint(1, 28), rng.randint(6, 20), rng.randint(0, 59)
        )
        channel = rng.choice(CHANNELS)
        if i % 23 == 7:
            mode = "misc"
            text = MISFILED_TICKETS[(i // 23) % len(MISFILED_TICKETS)]
        else:
            mode = modes[i % len(modes)]
            template = rng.choice(TICKET_TEMPLATES[mode])
            text = template.format(
                sid="SH25%05d" % (10000 + rng.randint(0, 399)),
                num=rng.randint(1, 240),
                num2=rng.randint(1, 240),
                street=rng.choice(STREETS),
            )
        resolution = rng.choice(RESOLUTIONS[mode])
        lines.append(json.dumps(
            {
                "ticket_id": tid,
                "created_at": created,
                "channel": channel,
                "text": text,
                "resolution": resolution,
            },
            ensure_ascii=False,
            sort_keys=True,
        ))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)
    random.seed(SEED)

    DATA.mkdir(parents=True, exist_ok=True)
    POLICIES.mkdir(parents=True, exist_ok=True)

    written = []

    p = DATA / "shipments_2024.csv"
    p.write_bytes(_build_2024(rng))
    written.append(p)

    p = DATA / "shipments_2025.csv"
    p.write_bytes(_build_2025(rng))
    written.append(p)

    for name in sorted(POLICY_DOCS):
        p = POLICIES / name
        p.write_bytes(POLICY_DOCS[name].encode("utf-8"))
        written.append(p)

    p = ROOT / "tickets.jsonl"
    p.write_bytes(_build_tickets(rng).encode("utf-8"))
    written.append(p)

    # Sanity check that pandas can at least see the 2024 file. The 2025 file is
    # not expected to load naively, which is the exercise.
    frame_2024 = pd.read_csv(DATA / "shipments_2024.csv")

    for path in written:
        print("wrote %s (%d bytes)" % (path.relative_to(ROOT.parent), path.stat().st_size))
    print("2024 export parsed naively: %d rows, %d columns" % frame_2024.shape)


if __name__ == "__main__":
    main()
