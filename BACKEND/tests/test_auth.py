import pytest

@pytest.mark.asyncio
async def test_register_and_login(client):
    # Register
    payload = {
        "email": "new@test.com",
        "password": "password123",
        "full_name": "New User",
        "phone_number": "+911234567890",
        "role": "user",
        "region": "assam",
        "state": "Assam",
        "district": "Kamrup"
    }
    res = await client.post("/auth/register", json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    assert "access_token" in data

    # Using stubbed auth, login endpoint can simply return stub
    login_res = await client.post("/auth/login/firebase", json={"token": "dummy"})
    assert login_res.status_code == 200
    assert login_res.json()["data"]["email"] == "test@example.com"
