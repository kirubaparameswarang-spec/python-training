import os
import requests
import hmac
import hashlib
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

if not API_KEY or not WEBHOOK_SECRET:
    raise ValueError("API_KEY and WEBHOOK_SECRET must be set in .env")

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session = requests.Session()
session.mount("https://", adapter)
url = "https://jsonplaceholder.typicode.com/users"

response = session.get(url)
response.raise_for_status()

users = response.json()

for user in users[:1]:
    print(f"{user.get('id')} - {user.get('name')} - {user.get('email')}")

first_user = users[0]

address = first_user.get("address", {})
geo = address.get("geo", {})
company = first_user.get("company", {})

city = address.get("city", "Not Available")
zip_code = "Not Available"

print(
    f"City: {city} | "
    f"Zip: {zip_code}" 
)

def verify_signature(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)

payload = '{"event": "payment_success"}'

valid_signature = hmac.new(
    WEBHOOK_SECRET.encode("utf-8"),
    payload.encode("utf-8"),
    hashlib.sha256
).hexdigest()

tampered_payload = '{"event": "payment_failed"}'

wrong_secret = "incorrect_secret"

valid = verify_signature(
    payload, valid_signature, WEBHOOK_SECRET
)

tampered = verify_signature(
    tampered_payload, valid_signature, WEBHOOK_SECRET
)

wrong = verify_signature(
    payload, valid_signature, wrong_secret
)

print(
    f"Valid: {valid} | "
    f"Tampered: {tampered} | "
    f"Wrong secret: {wrong}"
)
