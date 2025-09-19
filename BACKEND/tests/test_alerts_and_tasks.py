import pytest

@pytest.mark.asyncio
async def test_create_and_process_alert(client):
    headers = {"Authorization": "Bearer faketoken"}
    payload = {
        "alert_type": "general",
        "severity": "low",
        "title": "Test Alert",
        "message": "This is a test",
        "location": {"latitude": 26.2, "longitude": 91.7},
        "affected_radius_km": 1.0,
        "target_audience": ["user"]
    }
    res = await client.post("/alerts/create", json=payload, headers=headers)
    assert res.status_code == 200

    # The Celery task should run eagerly; test status
    task_res = await client.get("/alerts/active", headers=headers)
    assert task_res.status_code == 200
    assert isinstance(task_res.json()["data"], list)
