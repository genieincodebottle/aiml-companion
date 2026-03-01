"""
End-to-End Test Suite for Content Moderation API.

Tests 63 scenarios across 13 categories covering the complete user journey:
health, auth, content submission, stories, moderator review, HITL,
appeals, analytics, admin, profile, observability, password management, and logout.

Usage:
    python tests/test_e2e.py

Requirements:
    - Backend server running on localhost:8000
    - Demo users initialized (python scripts/initialize_users.py)
"""

import requests
import json
import time
import sys
import random
import string

BASE_URL = "http://localhost:8000"

# Test counters
passed = 0
failed = 0
total = 0
results_by_category = {}

# Shared state across tests
tokens = {}
created_content_ids = []
created_story_ids = []
created_appeal_ids = []
test_username = None
test_password = None


def test(category, name, method, endpoint, expected_status=200, json_data=None, headers=None, check_fn=None):
    """Run a single test case."""
    global passed, failed, total

    total += 1
    url = f"{BASE_URL}{endpoint}"

    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            resp = requests.post(url, json=json_data, headers=headers, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, json=json_data, headers=headers, timeout=30)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=30)
        else:
            print(f"  [FAIL] {name} - Unknown method: {method}")
            failed += 1
            return None

        status_ok = resp.status_code == expected_status

        if status_ok:
            if check_fn:
                try:
                    data = resp.json()
                    check_result = check_fn(data)
                    if check_result:
                        print(f"  [PASS] {method:6s} {endpoint:45s} {name}")
                        passed += 1
                        return data
                    else:
                        print(f"  [FAIL] {name} - Check function failed")
                        failed += 1
                        return None
                except Exception as e:
                    print(f"  [FAIL] {name} - Check error: {e}")
                    failed += 1
                    return None
            else:
                print(f"  [PASS] {method:6s} {endpoint:45s} {name}")
                passed += 1
                try:
                    return resp.json()
                except Exception:
                    return resp.text
        else:
            print(f"  [FAIL] {name} - Expected {expected_status}, got {resp.status_code}")
            try:
                print(f"         Response: {resp.text[:200]}")
            except Exception:
                pass
            failed += 1
            return None

    except requests.ConnectionError:
        print(f"  [FAIL] {name} - Connection refused (is the server running?)")
        failed += 1
        return None
    except Exception as e:
        print(f"  [FAIL] {name} - Error: {e}")
        failed += 1
        return None


def auth_header(role="user"):
    """Get auth header for a role."""
    token = tokens.get(role)
    if token:
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}


def run_category(name, test_fn):
    """Run a test category and track results."""
    global results_by_category
    before = passed
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    test_fn()
    cat_passed = passed - before
    results_by_category[name] = cat_passed


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1: Health & Server Status (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_health():
    test("Health", "Health check",
         "GET", "/health",
         check_fn=lambda d: d.get("status") == "healthy")

    test("Health", "Root endpoint",
         "GET", "/",
         check_fn=lambda d: "version" in d)

    test("Health", "ML status",
         "GET", "/api/ml/status",
         check_fn=lambda d: "ml_enabled" in d)

    test("Health", "API docs accessible",
         "GET", "/docs")


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2: Authentication (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_auth():
    global test_username, test_password

    # Login as user
    data = test("Auth", "Login as user (raj)",
                "POST", "/api/auth/login",
                json_data={"username": "raj", "password": "test@123"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["user"] = data["token"]

    # Login as moderator
    data = test("Auth", "Login as moderator",
                "POST", "/api/auth/login",
                json_data={"username": "moderator1", "password": "mod@123"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["moderator"] = data["token"]

    # Login as senior moderator
    data = test("Auth", "Login as senior moderator",
                "POST", "/api/auth/login",
                json_data={"username": "senior_mod", "password": "senior@123"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["senior_mod"] = data["token"]

    # Login as analyst
    data = test("Auth", "Login as analyst",
                "POST", "/api/auth/login",
                json_data={"username": "analyst", "password": "analyst@123"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["analyst"] = data["token"]

    # Login as policy specialist
    data = test("Auth", "Login as policy specialist",
                "POST", "/api/auth/login",
                json_data={"username": "policy_expert", "password": "policy@123"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["policy"] = data["token"]

    # Login as admin
    data = test("Auth", "Login as admin",
                "POST", "/api/auth/login",
                json_data={"username": "admin", "password": "admin@123"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["admin"] = data["token"]

    # Get current user
    test("Auth", "Get current user",
         "GET", "/api/auth/current-user",
         headers=auth_header("user"),
         check_fn=lambda d: d.get("username") == "raj")

    # Login with wrong password (should fail)
    test("Auth", "Login with wrong password (401)",
         "POST", "/api/auth/login",
         expected_status=401,
         json_data={"username": "raj", "password": "wrong_password"})

    # Register new test user
    suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
    test_username = f"e2e_test_{suffix}"
    test_password = "e2etest@123"
    data = test("Auth", "Register new user",
                "POST", "/api/auth/register",
                json_data={"username": test_username, "password": test_password, "full_name": "E2E Test User", "email": f"{test_username}@test.com"},
                check_fn=lambda d: d.get("success") is True)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3: Content Submission (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_content():
    # Submit clean content through full pipeline
    data = test("Content", "Submit clean content (full pipeline)",
                "POST", "/api/stories/submit",
                headers=auth_header("user"),
                json_data={"title": "E2E Test Story", "content_text": "Machine learning is transforming how we build software. Neural networks can now recognize patterns in complex datasets with remarkable accuracy. This technology will continue to evolve and improve over time.", "content_type": "story"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        created_story_ids.append(data.get("story_id", ""))
        created_content_ids.append(data.get("content_id", ""))

    # Submit toxic content via direct content endpoint
    data = test("Content", "Submit toxic content (direct endpoint)",
                "POST", "/api/content/submit",
                headers=auth_header("user"),
                json_data={"content_text": "You are a terrible disgusting person and I hate everything about you. Go away and never come back you worthless loser.", "content_type": "comment", "user_id": "e2e_test"},
                check_fn=lambda d: "content_id" in d)
    if data:
        created_content_ids.append(data.get("content_id", ""))

    # Get content by ID
    if created_content_ids:
        test("Content", "Get content by ID",
             "GET", f"/api/content/{created_content_ids[0]}",
             headers=auth_header("moderator"),
             check_fn=lambda d: d.get("content_id") == created_content_ids[0])

    # Get all content
    test("Content", "Get all content (paginated)",
         "GET", "/api/content?limit=5&offset=0",
         headers=auth_header("moderator"),
         check_fn=lambda d: "content" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4: Stories (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_stories():
    # View all stories
    test("Stories", "Get all stories",
         "GET", "/api/stories",
         headers=auth_header("user"),
         check_fn=lambda d: "stories" in d)

    # Get user's stories
    test("Stories", "Get user stories",
         "GET", "/api/stories/user/93",
         headers=auth_header("user"),
         check_fn=lambda d: isinstance(d, dict))

    # View single story
    if created_story_ids:
        test("Stories", "Get single story",
             "GET", f"/api/stories/{created_story_ids[0]}",
             headers=auth_header("user"),
             check_fn=lambda d: d.get("story_id") == created_story_ids[0])

        # Add good comment (fast mode)
        data = test("Stories", "Add good comment (fast mode)",
                    "POST", f"/api/stories/{created_story_ids[0]}/comments",
                    headers=auth_header("user"),
                    json_data={"content_text": "Great story! Really enjoyed reading this.", "content_type": "story_comment"},
                    check_fn=lambda d: d.get("success") is True)

        # Add toxic comment (should be removed)
        data = test("Stories", "Add toxic comment (fast mode, should remove)",
                    "POST", f"/api/stories/{created_story_ids[0]}/comments",
                    headers=auth_header("user"),
                    json_data={"content_text": "You are stupid and I hate you, go away loser", "content_type": "story_comment"},
                    check_fn=lambda d: d.get("success") is True)
        if data:
            created_content_ids.append(data.get("content_id", ""))

        # Get story comments
        test("Stories", "Get story comments",
             "GET", f"/api/stories/{created_story_ids[0]}/comments",
             headers=auth_header("user"),
             check_fn=lambda d: "comments" in d)

    # Get pending stories
    test("Stories", "Get pending stories",
         "GET", "/api/stories/pending",
         headers=auth_header("moderator"),
         check_fn=lambda d: "stories" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5: Moderator Review (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_moderator():
    # Pending content
    test("Moderator", "Get pending content",
         "GET", "/api/content/pending",
         headers=auth_header("moderator"),
         check_fn=lambda d: "content" in d)

    # All content
    test("Moderator", "Get all content",
         "GET", "/api/content/all",
         headers=auth_header("moderator"),
         check_fn=lambda d: "content" in d)

    # Pending comments
    test("Moderator", "Get pending comments",
         "GET", "/api/stories/comments/pending",
         headers=auth_header("moderator"),
         check_fn=lambda d: "comments" in d)

    # Content with pagination
    test("Moderator", "Get content paginated",
         "GET", "/api/content?limit=3&offset=0",
         headers=auth_header("moderator"),
         check_fn=lambda d: "content" in d)

    # Get specific content for review
    if created_content_ids:
        test("Moderator", "Get content detail for review",
             "GET", f"/api/content/{created_content_ids[0]}",
             headers=auth_header("moderator"),
             check_fn=lambda d: "content_id" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6: HITL Queue (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_hitl():
    # HITL queue
    test("HITL", "Get HITL queue",
         "GET", "/api/hitl/queue",
         headers=auth_header("senior_mod"),
         check_fn=lambda d: "queue" in d)

    # HITL config
    test("HITL", "Get HITL config",
         "GET", "/api/hitl/config",
         headers=auth_header("senior_mod"),
         check_fn=lambda d: "config" in d)

    # HITL queue as admin
    test("HITL", "Get HITL queue (admin)",
         "GET", "/api/hitl/queue",
         headers=auth_header("admin"),
         check_fn=lambda d: "total_pending" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 7: Appeals (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_appeals():
    # Find a removed content ID for appeal
    removed_id = None
    for cid in created_content_ids:
        if cid:
            try:
                resp = requests.get(f"{BASE_URL}/api/content/{cid}", headers=auth_header("moderator"), timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("current_status") == "removed" or data.get("moderation_action") == "removed":
                        removed_id = cid
                        break
            except Exception:
                pass

    # Submit appeal
    if removed_id:
        data = test("Appeals", "Submit appeal for removed content",
                    "POST", "/api/appeals/submit",
                    headers=auth_header("user"),
                    json_data={"content_id": removed_id, "user_id": "93", "appeal_reason": "E2E test: I believe this was incorrectly flagged."},
                    check_fn=lambda d: d.get("success") is True)
        if data:
            created_appeal_ids.append(data.get("appeal_id", ""))

    # Submit appeal (use first content id as fallback)
    appeal_content_id = removed_id or (created_content_ids[0] if created_content_ids else None)
    if appeal_content_id:
        data = test("Appeals", "Submit appeal",
                    "POST", "/api/appeals/submit",
                    headers=auth_header("user"),
                    json_data={"content_id": appeal_content_id, "user_id": "93", "appeal_reason": "E2E test: I believe this was incorrectly flagged."},
                    check_fn=lambda d: d.get("success") is True or "appeal_id" in d)

    # Get pending appeals
    test("Appeals", "Get pending appeals",
         "GET", "/api/appeals/pending",
         headers=auth_header("policy"),
         check_fn=lambda d: "appeals" in d)

    # Get all appeals
    test("Appeals", "Get all appeals",
         "GET", "/api/appeals/all",
         headers=auth_header("policy"),
         check_fn=lambda d: "appeals" in d)

    # Get appeals for content
    if appeal_content_id:
        test("Appeals", "Get appeals for content",
             "GET", f"/api/appeals/content/{appeal_content_id}",
             headers=auth_header("policy"),
             check_fn=lambda d: isinstance(d, (dict, list)))

    # Process appeal through AI
    if appeal_content_id:
        data = test("Appeals", "Process appeal through AI agent",
                    "POST", "/api/content/appeal",
                    headers=auth_header("policy"),
                    json_data={"content_id": appeal_content_id, "user_id": "93", "appeal_reason": "E2E test appeal processing"},
                    check_fn=lambda d: "decision" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 8: Analytics (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_analytics():
    test("Analytics", "Get statistics",
         "GET", "/api/statistics",
         headers=auth_header("admin"),
         check_fn=lambda d: "database" in d)

    test("Analytics", "Get analytics metrics",
         "GET", "/api/analytics/metrics",
         headers=auth_header("analyst"),
         check_fn=lambda d: "total_submissions" in d)

    test("Analytics", "Get agent performance",
         "GET", "/api/analytics/agent-performance",
         headers=auth_header("analyst"),
         check_fn=lambda d: "agents" in d)

    test("Analytics", "Get learning analytics",
         "GET", "/api/analytics/learning",
         headers=auth_header("analyst"),
         check_fn=lambda d: isinstance(d, dict))

    test("Analytics", "Get decision history",
         "GET", "/api/analytics/decisions",
         headers=auth_header("analyst"),
         check_fn=lambda d: "decisions" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 9: Admin (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_admin():
    test("Admin", "Get all users",
         "GET", "/api/auth/users",
         headers=auth_header("admin"),
         check_fn=lambda d: "users" in d)

    test("Admin", "Get audit log",
         "GET", "/api/auth/audit-log",
         headers=auth_header("admin"),
         check_fn=lambda d: "logs" in d)

    test("Admin", "Get moderator stats",
         "GET", "/api/auth/moderator-stats",
         headers=auth_header("admin"),
         check_fn=lambda d: "moderators" in d)

    test("Admin", "Get specific moderator stats",
         "GET", "/api/auth/moderator-stats/96",
         headers=auth_header("admin"),
         check_fn=lambda d: isinstance(d, dict))

    test("Admin", "Get specific user details",
         "GET", "/api/auth/users/93",
         headers=auth_header("admin"),
         check_fn=lambda d: isinstance(d, dict))

    test("Admin", "Get appeal trends",
         "GET", "/api/analytics/appeal-trends",
         headers=auth_header("admin"),
         check_fn=lambda d: isinstance(d, dict))


# ═══════════════════════════════════════════════════════════════════════════════
# Category 10: Profile & Reputation (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_profile():
    test("Profile", "Get user profile",
         "GET", "/api/users/93",
         headers=auth_header("admin"),
         check_fn=lambda d: "username" in d or "user_id" in d)

    test("Profile", "Get user reputation",
         "GET", "/api/users/93/reputation",
         headers=auth_header("admin"),
         check_fn=lambda d: "reputation_score" in d)

    test("Profile", "Get user history",
         "GET", "/api/users/93/history",
         headers=auth_header("admin"),
         check_fn=lambda d: isinstance(d, dict))

    # Login as test user and check profile access
    data = test("Profile", "Login as test user",
                "POST", "/api/auth/login",
                json_data={"username": test_username, "password": test_password},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["test_user"] = data["token"]

    test("Profile", "Get current user profile",
         "GET", "/api/auth/current-user",
         headers=auth_header("test_user") if "test_user" in tokens else auth_header("user"),
         check_fn=lambda d: "username" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 11: Observability (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_observability():
    test("Observability", "Observability health",
         "GET", "/api/observability/health",
         headers=auth_header("admin"),
         check_fn=lambda d: d.get("status") == "healthy")

    test("Observability", "Observability dashboard",
         "GET", "/api/observability/dashboard",
         headers=auth_header("admin"),
         check_fn=lambda d: "recent_logs" in d)

    test("Observability", "Get logs (filtered by level)",
         "GET", "/api/observability/logs?level=INFO&limit=10",
         headers=auth_header("admin"),
         check_fn=lambda d: "logs" in d)

    test("Observability", "Get performance metrics",
         "GET", "/api/observability/metrics?time_window_minutes=60",
         headers=auth_header("admin"),
         check_fn=lambda d: "metrics" in d)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 12: Password Management (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_password():
    if "test_user" not in tokens:
        # Login as test user first
        data = test("Password", "Login as test user",
                    "POST", "/api/auth/login",
                    json_data={"username": test_username, "password": test_password},
                    check_fn=lambda d: d.get("success") is True)
        if data:
            tokens["test_user"] = data["token"]

    # Change password
    test("Password", "Change password",
         "PUT", "/api/auth/password",
         headers=auth_header("test_user"),
         json_data={"current_password": test_password, "new_password": "newpass@456"},
         check_fn=lambda d: d.get("success") is True)

    # Login with new password
    data = test("Password", "Login with new password",
                "POST", "/api/auth/login",
                json_data={"username": test_username, "password": "newpass@456"},
                check_fn=lambda d: d.get("success") is True)
    if data:
        tokens["test_user"] = data["token"]

    # Restore original password
    test("Password", "Restore original password",
         "PUT", "/api/auth/password",
         headers=auth_header("test_user"),
         json_data={"current_password": "newpass@456", "new_password": test_password},
         check_fn=lambda d: d.get("success") is True)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 13: Logout (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════
def test_logout():
    test("Logout", "Logout user",
         "POST", "/api/auth/logout",
         headers=auth_header("user"),
         check_fn=lambda d: d.get("success") is True)

    test("Logout", "Logout admin",
         "POST", "/api/auth/logout",
         headers=auth_header("admin"),
         check_fn=lambda d: d.get("success") is True)

    # Verify logged out user can't access protected endpoints
    test("Logout", "Protected endpoint after logout (401)",
         "GET", "/api/auth/current-user",
         expected_status=401,
         headers=auth_header("user"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global passed, failed, total

    start_time = time.time()

    print("\n" + "=" * 70)
    print("  Content Moderation API - End-to-End Test Suite")
    print("  Server: " + BASE_URL)
    print("=" * 70)

    # Verify server is running
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("\n[ERROR] Server returned non-200 status. Is it running?")
            sys.exit(1)
    except requests.ConnectionError:
        print("\n[ERROR] Cannot connect to server at " + BASE_URL)
        print("Start the server first: python main.py")
        sys.exit(1)

    # Run all 13 categories
    categories = [
        ("1. Health & Server Status", test_health),
        ("2. Authentication", test_auth),
        ("3. Content Submission", test_content),
        ("4. Stories", test_stories),
        ("5. Moderator Review", test_moderator),
        ("6. HITL Queue", test_hitl),
        ("7. Appeals", test_appeals),
        ("8. Analytics", test_analytics),
        ("9. Admin", test_admin),
        ("10. Profile & Reputation", test_profile),
        ("11. Observability", test_observability),
        ("12. Password Management", test_password),
        ("13. Logout", test_logout),
    ]

    for name, test_fn in categories:
        run_category(name, test_fn)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)

    for cat_name, cat_passed in results_by_category.items():
        print(f"  {cat_name:45s} {cat_passed} passed")

    print(f"\n  {'TOTAL':45s} {passed}/{total} passed")
    print(f"  {'TIME':45s} {elapsed:.1f}s")

    if failed > 0:
        print(f"\n  [WARNING] {failed} test(s) FAILED")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f"\n  [OK] All {total} tests PASSED")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
