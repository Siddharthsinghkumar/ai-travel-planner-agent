# tools/booking_handoff.py
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, JSON, DateTime
from agents.database import Base, SessionLocal, engine

# Booking model attached to agents.database.Base so init_db will pick it up
class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, nullable=False)        # HELD | CONFIRMED | CANCELLED | EXPIRED
    flight = Column(JSON, nullable=False)
    passenger = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

# Ensure the table exists
def ensure_tables():
    Base.metadata.create_all(bind=engine)

ensure_tables()

# Core functions
def hold_booking(*, flight: dict, passenger: dict | None = None, hold_minutes: int = 15) -> int:
    """
    Create a temporary booking hold. Returns booking id.
    """
    db = SessionLocal()
    booking = Booking(
        status="HELD",
        flight=flight,
        passenger=passenger,
        expires_at=datetime.utcnow() + timedelta(minutes=hold_minutes)
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    db.close()
    return booking.id

def get_booking(booking_id: int):
    db = SessionLocal()
    b = db.get(Booking, booking_id)
    db.close()
    return None if not b else {
        "id": b.id,
        "status": b.status,
        "flight": b.flight,
        "passenger": b.passenger,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "expires_at": b.expires_at.isoformat() if b.expires_at else None
    }

def confirm_booking(booking_id: int) -> bool:
    db = SessionLocal()
    b = db.get(Booking, booking_id)
    if not b:
        db.close()
        return False
    # expire if already past expiry
    if b.expires_at and b.expires_at < datetime.utcnow():
        b.status = "EXPIRED"
        db.commit()
        db.close()
        return False
    if b.status != "HELD":
        db.close()
        return False
    b.status = "CONFIRMED"
    db.commit()
    db.close()
    return True

def cancel_booking(booking_id: int) -> bool:
    db = SessionLocal()
    b = db.get(Booking, booking_id)
    if not b:
        db.close()
        return False
    if b.status in ("CONFIRMED", "EXPIRED"):
        db.close()
        return False
    b.status = "CANCELLED"
    db.commit()
    db.close()
    return True

def expire_bookings() -> int:
    """
    Mark all HELD bookings past their expiry as EXPIRED.
    Returns number expired.
    """
    db = SessionLocal()
    now = datetime.utcnow()
    q = db.query(Booking).filter(Booking.status == "HELD", Booking.expires_at < now)
    expired = q.all()
    count = len(expired)
    for b in expired:
        b.status = "EXPIRED"
    db.commit()
    db.close()
    return count

# Hand-off URL builders (safe deep-links, no scraping)
import urllib.parse

AIRLINE_BOOKING_URLS = {
    "air india": "https://www.airindia.in/booking",
    "indigo": "https://www.goindigo.in",
    "vistara": "https://www.airvistara.com",
    # add more if you need
}

def build_booking_handoff_url(*, airline: str, origin: str, destination: str, depart_date: str, return_date: str | None = None, passengers: int = 1) -> str:
    airline_l = (airline or "").lower()
    base = None
    for k in AIRLINE_BOOKING_URLS:
        if k in airline_l:
            base = AIRLINE_BOOKING_URLS[k]
            break
    if base:
        params = {
            "origin": origin,
            "destination": destination,
            "departDate": depart_date,
            "tripType": "R" if return_date else "O",
            "pax": passengers
        }
        if return_date:
            params["returnDate"] = return_date
        return f"{base}?{urllib.parse.urlencode(params)}"
    # fallback to Google Flights deep link
    date_part = depart_date if not return_date else f"{depart_date}/{return_date}"
    return f"https://www.google.com/travel/flights?hl=en#flt={origin}.{destination}.{date_part};c:INR;e:1;sd:1;t:f"
