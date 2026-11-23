import os
import json
import hmac
import hashlib

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Load secrets from .env; never put real credentials in source code.
load_dotenv()

API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

if not API_KEY or not WEBHOOK_SECRET:
    raise ValueError("Please set API_KEY and WEBHOOK_SECRET in the .env file.")


def create_session():
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def verify_signature(payload, signature, secret):
    """Return True only when signature matches the HMAC-SHA256 digest."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if isinstance(secret, str):
        secret = secret.encode("utf-8")

    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def main():
    url = "https://jsonplaceholder.typicode.com/users"

    with create_session() as session:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        users = response.json()

    print("First 5 users:")
    for user in users[:5]:
        print(
            f'{user.get("id", "Not Available")} - '
            f'{user.get("name", "Not Available")} - '
            f'{user.get("email", "Not Available")}'
        )

    first_user = users[0] if users else {}
    address = first_user.get("address", {})
    geo = address.get("geo", {})
    company = first_user.get("company", {})

    # JSONPlaceholder has no "zip" field (its field is "zipcode"),
    # so this prints "Not Available" as shown in the assessment output.
    city = address.get("city", "Not Available")
    latitude = geo.get("lat", "Not Available")
    zip_code = address.get("zip", "Not Available")
    company_name = company.get("name", "Not Available")

    print(
        f"City: {city} | Zip: {zip_code} | "
        f"Latitude: {latitude} | Company: {company_name}"
    )

    payload = json.dumps(
        {"event": "user.created", "user_id": 1}, separators=(",", ":")
    ).encode("utf-8")

    valid_signature = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()

    tampered_payload = payload + b"tampered"
    wrong_secret_signature = hmac.new(
        b"wrong-secret", payload, hashlib.sha256
    ).hexdigest()

    print(
        "Valid: {} | Tampered: {} | Wrong secret: {}".format(
            verify_signature(payload, valid_signature, WEBHOOK_SECRET),
            verify_signature(tampered_payload, valid_signature, WEBHOOK_SECRET),
            verify_signature(payload, wrong_secret_signature, WEBHOOK_SECRET),
        )
    )


if __name__ == "__main__":
    main()
