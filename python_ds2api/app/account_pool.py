"""
Account pool management for DS2API.
Handles account selection, token refresh, and concurrency control.
"""
import asyncio
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque

from .config import config_store, AccountConfig
from .deepseek_client import deepseek_client, RequestAuth


@dataclass
class AccountSlot:
    """Represents an account with its current state."""
    account_id: str
    account: AccountConfig
    token: Optional[str] = None
    in_flight: int = 0
    last_used: float = 0.0
    token_refreshed_at: float = 0.0
    is_valid: bool = True


class AccountPool:
    """Manages account selection and concurrency."""

    def __init__(self):
        self._slots: Dict[str, AccountSlot] = {}
        self._waiters: deque = deque()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the account pool with configured accounts."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            for acc in config_store.config.accounts:
                account_id = acc.email or acc.mobile
                if not account_id:
                    continue

                # Check for stored token
                stored_token = config_store.get_account_token(account_id)

                self._slots[account_id] = AccountSlot(
                    account_id=account_id,
                    account=acc,
                    token=stored_token,
                    is_valid=True
                )

            self._initialized = True

    async def login_all(self):
        """Login to all accounts and refresh tokens."""
        await self.initialize()

        tasks = []
        for slot in self._slots.values():
            tasks.append(self._login_account(slot))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        print(f"Logged in {success_count}/{len(self._slots)} accounts")

    async def _login_account(self, slot: AccountSlot) -> bool:
        """Login to a single account."""
        try:
            token = await deepseek_client.login(slot.account)
            if token:
                slot.token = token
                slot.token_refreshed_at = time.time()
                slot.is_valid = True
                config_store.set_account_token(slot.account_id, token)
                return True
        except Exception as e:
            print(f"Login failed for {slot.account_id}: {e}")
            slot.is_valid = False

        return False

    async def acquire(self, target_account: Optional[str] = None) -> Optional[RequestAuth]:
        """
        Acquire an account slot for a request.
        Returns RequestAuth with the account token.
        """
        await self.initialize()

        async with self._lock:
            # If target account specified, try to use it
            if target_account and target_account in self._slots:
                slot = self._slots[target_account]
                if slot.is_valid and slot.token:
                    if slot.in_flight < config_store.config.runtime.account_max_inflight:
                        slot.in_flight += 1
                        slot.last_used = time.time()
                        return RequestAuth(
                            deepseek_token=slot.token,
                            account_id=slot.account_id,
                            use_config_token=True
                        )

            # Otherwise, find the best available account
            best_slot = None
            best_inflight = float('inf')

            for slot in self._slots.values():
                if not slot.is_valid or not slot.token:
                    continue

                if slot.in_flight < config_store.config.runtime.account_max_inflight:
                    if slot.in_flight < best_inflight:
                        best_inflight = slot.in_flight
                        best_slot = slot

            if best_slot:
                best_slot.in_flight += 1
                best_slot.last_used = time.time()
                return RequestAuth(
                    deepseek_token=best_slot.token,
                    account_id=best_slot.account_id,
                    use_config_token=True
                )

        return None

    def release(self, auth: RequestAuth):
        """Release an account slot after request completion."""
        if auth.account_id in self._slots:
            slot = self._slots[auth.account_id]
            if slot.in_flight > 0:
                slot.in_flight -= 1

    async def refresh_token(self, account_id: str) -> bool:
        """Refresh token for a specific account."""
        if account_id not in self._slots:
            return False

        slot = self._slots[account_id]
        return await self._login_account(slot)

    async def refresh_stale_tokens(self):
        """Refresh tokens that haven't been updated in a while."""
        refresh_interval = config_store.config.runtime.token_refresh_interval_hours * 3600
        current_time = time.time()

        for slot in self._slots.values():
            if current_time - slot.token_refreshed_at > refresh_interval:
                await self._login_account(slot)

    def get_status(self) -> List[Dict]:
        """Get status of all accounts."""
        return [
            {
                "account_id": slot.account_id,
                "in_flight": slot.in_flight,
                "is_valid": slot.is_valid,
                "has_token": bool(slot.token),
                "last_used": slot.last_used
            }
            for slot in self._slots.values()
        ]


# Global account pool instance
account_pool = AccountPool()
