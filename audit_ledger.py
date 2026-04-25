# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Cryptographic Audit Ledger for VitalChain-Env.

Implements an immutable, hash-chained audit trail for every biological
resource allocation, transport, and receipt event. This module provides:

1. BlockchainLedger: A mock distributed ledger that records SHA-256
   hashed events in an append-only chain, ensuring tamper detection.

2. Digital Birth Certificate: Each organ receives a cryptographic
   identity at harvest time, linked to its NOTTO registry entry.

3. Waitlist Verification: Before any organ allocation, the system
   verifies the recipient exists on the centralized NOTTO waitlist
   and that the organ's provenance chain is unbroken.

4. Black Market Prevention: Any attempt to allocate an organ without
   a valid birth certificate or to a non-waitlisted patient triggers
   an alert and blocks the transfer.

In production, this would integrate with:
  - Hyperledger Fabric or Ethereum L2 for actual DLT
  - NOTTO (National Organ & Tissue Transplant Organisation) API
  - eRaktKosh for blood product chain-of-custody
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# ── Event Types ──────────────────────────────────────────────────────────────

class AuditEventType(Enum):
    """Types of events recorded on the audit ledger."""
    ORGAN_HARVESTED     = "organ_harvested"
    RESOURCE_CREATED    = "resource_created"
    ALLOCATION_APPROVED = "allocation_approved"
    TRANSPORT_INITIATED = "transport_initiated"
    TRANSPORT_HANDOFF   = "transport_handoff"
    TRANSPORT_RECEIVED  = "transport_received"
    TRANSPLANT_COMPLETE = "transplant_complete"
    RESOURCE_EXPIRED    = "resource_expired"
    RESOURCE_DISCARDED  = "resource_discarded"
    VERIFICATION_FAILED = "verification_failed"
    WAITLIST_CHECK      = "waitlist_check"
    TAMPER_DETECTED     = "tamper_detected"


# ── Ledger Entry ─────────────────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    """A single immutable entry in the audit chain."""
    index: int
    timestamp: float
    event_type: str
    resource_id: str
    actor_id: str                    # hospital_id or system
    details: Dict = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Generate SHA-256 hash of this entry's contents."""
        payload = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "resource_id": self.resource_id,
            "actor_id": self.actor_id,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# ── Digital Birth Certificate ────────────────────────────────────────────────

@dataclass
class DigitalBirthCertificate:
    """
    Cryptographic identity for an organ at the moment of harvest.
    
    This is the organ's "passport" — it must be verified at every
    handoff point. Any break in the chain flags the organ for
    immediate quarantine (black market prevention).
    """
    certificate_id: str
    resource_id: str
    notto_id: str                    # NOTTO registry reference
    organ_type: str
    blood_type: str
    hla_type: str
    donor_hospital_id: str
    harvest_timestamp: float
    max_ischemic_hours: float
    certificate_hash: str = ""       # SHA-256 of the certificate contents
    is_revoked: bool = False         # set True if tampering detected

    def compute_certificate_hash(self) -> str:
        """Generate the certificate's unique fingerprint."""
        payload = json.dumps({
            "certificate_id": self.certificate_id,
            "resource_id": self.resource_id,
            "notto_id": self.notto_id,
            "organ_type": self.organ_type,
            "blood_type": self.blood_type,
            "hla_type": self.hla_type,
            "donor_hospital_id": self.donor_hospital_id,
            "harvest_timestamp": self.harvest_timestamp,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# ── NOTTO Waitlist (Simulated) ───────────────────────────────────────────────

class NOTTOWaitlist:
    """
    Simulated NOTTO centralized organ waitlist.
    
    Every patient registered for organ transplant must appear on this list.
    Organs can ONLY be allocated to waitlisted patients. This is the primary
    black market prevention mechanism.
    """

    def __init__(self):
        self._waitlist: Dict[str, Dict] = {}
        self._allocation_log: List[Dict] = []

    def register_patient(
        self, patient_id: str, organ_needed: str,
        blood_type: str, hospital_id: str,
        urgency: int = 3, hla_type: str = ""
    ) -> str:
        """Register a patient on the NOTTO waitlist. Returns waitlist ID."""
        waitlist_id = f"NOTTO-WL-{hashlib.md5(patient_id.encode()).hexdigest()[:8].upper()}"
        self._waitlist[patient_id] = {
            "waitlist_id": waitlist_id,
            "patient_id": patient_id,
            "organ_needed": organ_needed,
            "blood_type": blood_type,
            "hospital_id": hospital_id,
            "urgency": urgency,
            "hla_type": hla_type,
            "registered_at": time.time(),
            "status": "active",
        }
        return waitlist_id

    def is_patient_waitlisted(self, patient_id: str, organ_type: str = "") -> bool:
        """Check if a patient is on the active waitlist for the given organ."""
        entry = self._waitlist.get(patient_id)
        if not entry:
            return False
        if entry["status"] != "active":
            return False
        if organ_type and entry["organ_needed"] != organ_type:
            return False
        return True

    def get_waitlist_entry(self, patient_id: str) -> Optional[Dict]:
        """Get full waitlist entry for a patient."""
        return self._waitlist.get(patient_id)

    def mark_allocated(self, patient_id: str, resource_id: str) -> None:
        """Mark a waitlist entry as allocated (organ assigned)."""
        if patient_id in self._waitlist:
            self._waitlist[patient_id]["status"] = "allocated"
            self._allocation_log.append({
                "patient_id": patient_id,
                "resource_id": resource_id,
                "allocated_at": time.time(),
            })

    @property
    def active_count(self) -> int:
        """Number of patients actively waiting."""
        return sum(1 for e in self._waitlist.values() if e["status"] == "active")


# ── Blockchain Audit Ledger ──────────────────────────────────────────────────

class BlockchainLedger:
    """
    Mock distributed ledger for organ/blood chain-of-custody.
    
    Every resource event is recorded as a hash-chained entry.
    The chain is append-only — any modification to a past entry
    would break the hash chain and trigger tamper detection.
    
    Key operations:
    1. record_event() — append a new entry to the chain
    2. verify_chain() — check integrity of the entire chain
    3. issue_birth_certificate() — create organ's digital identity
    4. verify_allocation() — check organ + patient against waitlist
    """

    def __init__(self):
        self._chain: List[LedgerEntry] = []
        self._certificates: Dict[str, DigitalBirthCertificate] = {}
        self._waitlist = NOTTOWaitlist()
        self._stats = {
            "total_events": 0,
            "allocations_verified": 0,
            "allocations_rejected": 0,
            "tamper_alerts": 0,
            "certificates_issued": 0,
            "certificates_revoked": 0,
        }
        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create the first block in the chain."""
        genesis = LedgerEntry(
            index=0,
            timestamp=time.time(),
            event_type="genesis",
            resource_id="SYSTEM",
            actor_id="VITALCHAIN",
            details={"message": "VitalChain Audit Ledger initialized"},
            previous_hash="0" * 64,
        )
        genesis.entry_hash = genesis.compute_hash()
        self._chain.append(genesis)

    def record_event(
        self,
        event_type: str,
        resource_id: str,
        actor_id: str,
        details: Dict = None,
    ) -> LedgerEntry:
        """
        Record a new event on the audit chain.
        
        Every event is cryptographically linked to the previous one.
        """
        if details is None:
            details = {}

        previous = self._chain[-1]
        entry = LedgerEntry(
            index=len(self._chain),
            timestamp=time.time(),
            event_type=event_type,
            resource_id=resource_id,
            actor_id=actor_id,
            details=details,
            previous_hash=previous.entry_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._chain.append(entry)
        self._stats["total_events"] += 1
        return entry

    def issue_birth_certificate(
        self,
        resource_id: str,
        notto_id: str,
        organ_type: str,
        blood_type: str,
        hla_type: str,
        donor_hospital_id: str,
        harvest_timestamp: float,
        max_ischemic_hours: float,
    ) -> DigitalBirthCertificate:
        """
        Issue a Digital Birth Certificate for an organ at harvest time.
        
        This creates the organ's cryptographic identity and records
        the issuance event on the ledger.
        """
        cert_id = f"DBC-{hashlib.sha256(f'{resource_id}:{notto_id}'.encode()).hexdigest()[:12].upper()}"

        cert = DigitalBirthCertificate(
            certificate_id=cert_id,
            resource_id=resource_id,
            notto_id=notto_id,
            organ_type=organ_type,
            blood_type=blood_type,
            hla_type=hla_type,
            donor_hospital_id=donor_hospital_id,
            harvest_timestamp=harvest_timestamp,
            max_ischemic_hours=max_ischemic_hours,
        )
        cert.certificate_hash = cert.compute_certificate_hash()
        self._certificates[resource_id] = cert
        self._stats["certificates_issued"] += 1

        # Record on ledger
        self.record_event(
            event_type=AuditEventType.ORGAN_HARVESTED.value,
            resource_id=resource_id,
            actor_id=donor_hospital_id,
            details={
                "certificate_id": cert_id,
                "certificate_hash": cert.certificate_hash,
                "organ_type": organ_type,
                "notto_id": notto_id,
                "max_ischemic_hours": max_ischemic_hours,
            },
        )
        return cert

    def verify_birth_certificate(self, resource_id: str) -> dict:
        """
        Verify an organ's Digital Birth Certificate.
        
        Returns verification result with pass/fail and reason.
        """
        cert = self._certificates.get(resource_id)
        if not cert:
            return {
                "verified": False,
                "reason": "No birth certificate found — organ provenance unknown",
                "alert_level": "CRITICAL",
            }

        if cert.is_revoked:
            return {
                "verified": False,
                "reason": "Certificate has been revoked — possible tampering",
                "alert_level": "CRITICAL",
            }

        # Recompute hash to check for tampering
        expected_hash = cert.compute_certificate_hash()
        if expected_hash != cert.certificate_hash:
            cert.is_revoked = True
            self._stats["certificates_revoked"] += 1
            self._stats["tamper_alerts"] += 1
            self.record_event(
                event_type=AuditEventType.TAMPER_DETECTED.value,
                resource_id=resource_id,
                actor_id="VERIFICATION_SYSTEM",
                details={"expected": expected_hash, "found": cert.certificate_hash},
            )
            return {
                "verified": False,
                "reason": "Certificate hash mismatch — TAMPER DETECTED",
                "alert_level": "CRITICAL",
            }

        return {
            "verified": True,
            "certificate_id": cert.certificate_id,
            "notto_id": cert.notto_id,
            "organ_type": cert.organ_type,
            "alert_level": "NONE",
        }

    def verify_allocation(
        self,
        resource_id: str,
        patient_id: str,
        receiving_hospital_id: str,
    ) -> dict:
        """
        Full allocation verification: birth certificate + waitlist check.
        
        This is the black market prevention gate. Both checks must pass:
        1. Organ must have a valid, untampered Digital Birth Certificate
        2. Patient must be on the active NOTTO waitlist for this organ type
        
        Returns:
            {
                "approved": bool,
                "certificate_verified": bool,
                "waitlist_verified": bool,
                "rejection_reasons": list,
                "ledger_entry_hash": str,
            }
        """
        result = {
            "approved": False,
            "certificate_verified": False,
            "waitlist_verified": False,
            "rejection_reasons": [],
            "ledger_entry_hash": "",
        }

        # Step 1: Verify birth certificate
        cert_check = self.verify_birth_certificate(resource_id)
        result["certificate_verified"] = cert_check["verified"]
        if not cert_check["verified"]:
            result["rejection_reasons"].append(cert_check["reason"])

        # Step 2: Verify patient is on waitlist
        cert = self._certificates.get(resource_id)
        organ_type = cert.organ_type if cert else ""
        waitlist_ok = self._waitlist.is_patient_waitlisted(patient_id, organ_type)
        result["waitlist_verified"] = waitlist_ok
        if not waitlist_ok:
            result["rejection_reasons"].append(
                f"Patient {patient_id} not on active NOTTO waitlist for {organ_type}"
            )

        # Final decision
        if result["certificate_verified"] and result["waitlist_verified"]:
            result["approved"] = True
            self._stats["allocations_verified"] += 1
            self._waitlist.mark_allocated(patient_id, resource_id)
            entry = self.record_event(
                event_type=AuditEventType.ALLOCATION_APPROVED.value,
                resource_id=resource_id,
                actor_id=receiving_hospital_id,
                details={
                    "patient_id": patient_id,
                    "organ_type": organ_type,
                    "certificate_id": cert.certificate_id if cert else "MISSING",
                },
            )
            result["ledger_entry_hash"] = entry.entry_hash
        else:
            self._stats["allocations_rejected"] += 1
            self.record_event(
                event_type=AuditEventType.VERIFICATION_FAILED.value,
                resource_id=resource_id,
                actor_id=receiving_hospital_id,
                details={
                    "patient_id": patient_id,
                    "rejection_reasons": result["rejection_reasons"],
                },
            )

        return result

    def record_transport_handoff(
        self,
        resource_id: str,
        from_hospital_id: str,
        to_hospital_id: str,
        route_type: str = "standard",
    ) -> LedgerEntry:
        """Record a transport handoff between hospitals."""
        return self.record_event(
            event_type=AuditEventType.TRANSPORT_INITIATED.value,
            resource_id=resource_id,
            actor_id=from_hospital_id,
            details={
                "from": from_hospital_id,
                "to": to_hospital_id,
                "route_type": route_type,
            },
        )

    def verify_chain_integrity(self) -> dict:
        """
        Verify the entire blockchain for tampering.
        
        Checks that every block's hash is valid and that the chain
        of previous_hash references is unbroken.
        
        Returns:
            {
                "valid": bool,
                "blocks_checked": int,
                "tampered_blocks": list,
            }
        """
        tampered = []

        for i in range(1, len(self._chain)):
            entry = self._chain[i]
            # Check hash integrity
            computed = entry.compute_hash()
            if computed != entry.entry_hash:
                tampered.append(i)
                continue
            # Check chain linkage
            if entry.previous_hash != self._chain[i - 1].entry_hash:
                tampered.append(i)

        if tampered:
            self._stats["tamper_alerts"] += len(tampered)

        return {
            "valid": len(tampered) == 0,
            "blocks_checked": len(self._chain),
            "tampered_blocks": tampered,
        }

    def get_resource_history(self, resource_id: str) -> List[dict]:
        """Get complete audit trail for a specific resource."""
        return [
            {
                "index": entry.index,
                "event_type": entry.event_type,
                "actor_id": entry.actor_id,
                "timestamp": entry.timestamp,
                "details": entry.details,
                "hash": entry.entry_hash[:16] + "...",
            }
            for entry in self._chain
            if entry.resource_id == resource_id
        ]

    @property
    def stats(self) -> dict:
        """Get audit ledger statistics."""
        return {
            **self._stats,
            "chain_length": len(self._chain),
            "active_certificates": sum(
                1 for c in self._certificates.values() if not c.is_revoked
            ),
        }

    @property
    def waitlist(self) -> NOTTOWaitlist:
        """Access the NOTTO waitlist."""
        return self._waitlist
