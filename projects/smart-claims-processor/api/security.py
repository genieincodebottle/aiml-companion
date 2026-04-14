"""JWT + bcrypt helpers and current-user dependency."""
from __future__ import annotations # for Python 3.10-3.11 compatibility, allows forward references in type hints without quotes

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException
import bcrypt
from jose import JWTError, jwt
from sqlmodel import Session, select

from api.db import User, get_session

SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = _decode(token)
    user = session.get(User, int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_role(*roles: str):
    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role in {roles}")
        return user
    return _dep


def user_to_dict(user: User) -> dict:
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role}


# Built-in accounts created on first startup so the HITL approval flow is
# testable end-to-end without manual registration. Passwords are for dev only.
SEED_USERS: list[dict] = [
    {
        "username": "admin",
        "email": "admin@example.com",
        "password": "admin123",
        "role": "admin",      # manages users, switches LLM provider
    },
    {
        "username": "reviewer1",
        "email": "reviewer1@example.com",
        "password": "review123",
        "role": "reviewer",   # approves HITL tickets, resumes paused claims
    },
    {
        "username": "reviewer2",
        "email": "reviewer2@example.com",
        "password": "review123",
        "role": "reviewer",   # second approver for parallel testing
    },
    {
        "username": "claimant",
        "email": "claimant@example.com",
        "password": "claim123",
        "role": "user",       # files claims, views own status
    },
]


def seed_admin(session: Session) -> None:
    """Seed all dev accounts (admin, reviewers, claimant). Idempotent."""
    created: list[str] = []
    for spec in SEED_USERS:
        existing = session.exec(select(User).where(User.username == spec["username"])).first()
        if existing:
            continue
        session.add(User(
            username=spec["username"],
            email=spec["email"],
            password_hash=hash_password(spec["password"]),
            role=spec["role"],
        ))
        created.append(f"{spec['username']} ({spec['role']})")
    if created:
        session.commit()
        import logging
        logging.getLogger(__name__).info("Seeded users: %s", ", ".join(created))
