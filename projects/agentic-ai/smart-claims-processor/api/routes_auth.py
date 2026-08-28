"""Auth routes: login, register, users CRUD, password update."""
from __future__ import annotations # for forward references in type hints (Python 3.10+)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from api.db import User, get_session
from api.security import (
    create_token,
    get_current_user,
    hash_password,
    require_role,
    user_to_dict,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "user"


class PasswordUpdateRequest(BaseModel):
    current_password: str
    new_password: str


class UserUpdateRequest(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    role: str | None = None


@router.post("/login")
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == body.username)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user": user_to_dict(user), "token": create_token(user.id, user.username, user.role)}


@router.post("/register")
def register(body: RegisterRequest, session: Session = Depends(get_session)):
    if session.exec(select(User).where(User.username == body.username)).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if session.exec(select(User).where(User.email == body.email)).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role if body.role in ("user", "reviewer", "admin") else "user",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"user": user_to_dict(user), "token": create_token(user.id, user.username, user.role)}


@router.post("/logout")
def logout(user: User = Depends(get_current_user)):
    return {"ok": True}


@router.get("/current-user")
def current_user(user: User = Depends(get_current_user)):
    return {"user": user_to_dict(user)}


@router.get("/users")
def list_users(
    session: Session = Depends(get_session),
    _: User = Depends(require_role("admin")),
):
    users = session.exec(select(User)).all()
    return [user_to_dict(u) for u in users]


@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    body: UserUpdateRequest,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("admin")),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if body.username is not None:
        user.username = body.username
    if body.email is not None:
        user.email = body.email
    if body.role is not None:
        user.role = body.role
    session.add(user)
    session.commit()
    session.refresh(user)
    return user_to_dict(user)


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    session: Session = Depends(get_session),
    admin: User = Depends(require_role("admin")),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.delete(user)
    session.commit()
    return {"ok": True}


@router.put("/password")
def update_password(
    body: PasswordUpdateRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password incorrect")
    user.password_hash = hash_password(body.new_password)
    session.add(user)
    session.commit()
    return {"ok": True}
