# Address Correction Procedure

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
