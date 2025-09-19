import pytest

@pytest.mark.asyncio
async def test_upload_water_quality(client):
    headers = {"Authorization": "Bearer faketoken"}
    payload = {
        "sensor_id": "s1",
        "location": {"latitude": 26.2, "longitude": 91.7},
        "water_source": "tube_well",
        "ph_level": 7.0,
        "turbidity": 1.2
    }
    res = await client.post("/data/water-quality", json=payload, headers=headers)
    assert res.status_code == 200
    assert res.json()["data"]["document_id"]
