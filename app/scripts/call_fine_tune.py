import requests as rq

payload = {
    "dataset_uuid": "0ff775cd-86b5-49d3-a3a2-a0849bdf2802",
    "config_uuid": "2de36889-f9ce-4238-a0ce-69a4e434e5a4",
}

r = rq.post("http://0.0.0.0:8001/api/v1/fine_tune_stable_diffusion", json=payload)
print(r.content)
