# Northwind Freight Delivery Service Level Agreement

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
