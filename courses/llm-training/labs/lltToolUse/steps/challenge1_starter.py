# A BFCL-style checker for parsed tool calls. A call is {name: {param: value}}.
# An expected call is {name: {param: [allowed values]}}; "" in the list means the parameter may be left out.

def norm(s):
    for ch in " ,./-_*^":
        s = s.replace(ch, "")
    return s.lower()

def value_ok(value, allowed):
    for a in allowed:
        if isinstance(a, str) and isinstance(value, str) and norm(a) == norm(value):
            return True
        if not isinstance(a, str) and not isinstance(value, str) and a == value and type(a) is not bool and type(value) is not bool:
            return True
        if type(a) is bool and type(value) is bool and a == value:
            return True
    return False

def call_ok(call, expected, schema):
    # True when call matches expected: same name, required params present, no unknown params,
    # every left-out param allowed to be left out, every value allowed.
    raise NotImplementedError

def check(item, calls):
    # calls: a list of calls, [] for no call, None for a malformed reply.
    # Irrelevance items are right when no well-formed call comes out. Otherwise the calls must match the
    # expected list one to one, in any order.
    raise NotImplementedError

# Checks. Run the cell; it prints All checks passed when call_ok and check are right.
area = {"required": ["base", "height"], "properties": {"base": {}, "height": {}, "unit": {}}}
simple = {"category": "simple", "schemas": {"area": area},
          "expected": [{"area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}
assert check(simple, [{"area": {"base": 10, "height": 5}}]), "unit may be left out"
assert check(simple, [{"area": {"base": 10, "height": 5, "unit": "Units"}}]), "strings compare without case"
assert not check(simple, [{"area": {"base": 10, "height": 6}}]), "wrong value"
assert not check(simple, [{"area": {"base": 10}}]), "height is required"
assert not check(simple, [{"area": {"base": 10, "height": 5, "colour": "red"}}]), "colour is not a parameter"
assert not check(simple, [{"volume": {"base": 10, "height": 5}}]), "wrong function"
assert not check(simple, []) and not check(simple, None), "no call, or a malformed one, is wrong"
power = {"required": ["base", "exponent"], "properties": {"base": {}, "exponent": {}}}
parallel = {"category": "parallel", "schemas": {"power": power},
            "expected": [{"power": {"base": [2], "exponent": [3]}}, {"power": {"base": [3], "exponent": [5]}}]}
assert check(parallel, [{"power": {"base": 3, "exponent": 5}}, {"power": {"base": 2, "exponent": 3}}]), "any order"
assert not check(parallel, [{"power": {"base": 2, "exponent": 3}}]), "one call is missing"
assert not check(parallel, [{"power": {"base": 2, "exponent": 3}}, {"power": {"base": 2, "exponent": 3}}]), "one call may match one expected call only"
irrelevant = {"category": "irrelevance", "schemas": {"area": area}, "expected": None}
assert check(irrelevant, []) and check(irrelevant, None), "no well-formed call is right here"
assert not check(irrelevant, [{"area": {"base": 1, "height": 1}}]), "any call is wrong here"
assert not call_ok({"area": {"base": True, "height": 5}}, simple["expected"][0], area), "True is not 10"
print("All checks passed")
