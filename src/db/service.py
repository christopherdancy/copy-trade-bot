from decimal import Decimal
from datetime import datetime
from .database import DatabaseConnection
from .models import Signal, Position
from sqlalchemy.orm import Session
import logging

class DatabaseService:
    def __init__(self, run_id: str):
        self.db = DatabaseConnection()
        self.logger = logging.getLogger(__name__)
        self.run_id = run_id

    def save_signal(self, timestamp: datetime, mint: str, price: float, volume: float, signal_type: str) -> int:
        """Save a signal event to database"""
        try:
            session = self.db.get_session()
            signal = Signal(
                timestamp=timestamp,
                mint=mint,
                price=price,
                volume=volume,
                signal_type=signal_type,
                run_id=self.run_id
            )
            session.add(signal)
            session.commit()
            signal_id = signal.id
            session.close()
            return signal_id
        except Exception as e:
            self.logger.error(f"Error saving signal: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            raise

    def save_position(self, 
                      mint: str, 
                      entry_price: float, 
                      entry_time: datetime, 
                      position_size: float, 
                      token_amount: float, 
                      entry_signal_id: int = None):
        """Save a new position to database"""
        try:
            session = self.db.get_session()
            position = Position(
                mint=mint,
                entry_price=entry_price,
                entry_time=entry_time,
                position_size=position_size,
                token_amount=token_amount,
                is_active=True,
                entry_signal_id=entry_signal_id,
                run_id=self.run_id
            )
            session.add(position)
            session.commit()
            position_id = position.id
            session.close()
            return position_id
        except Exception as e:
            self.logger.error(f"Error saving position: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            raise

    def update_position_exit(self, mint: str, exit_price: float, exit_time: datetime, exit_signal_id: int):
        """Update position with exit information"""
        try:
            session = self.db.get_session()
            position = (
                session.query(Position)
                .filter(Position.mint == mint, Position.is_active == True)
                .first()
            )
            if position:
                position.exit_price = exit_price
                position.exit_time = exit_time
                position.is_active = False
                position.exit_signal_id = exit_signal_id
                session.commit()
            session.close()
        except Exception as e:
            self.logger.error(f"Error updating position exit: {str(e)}")
            if 'session' in locals():
                session.rollback()
                session.close()
            raise

    def get_active_positions(self):
        """Get all active positions"""
        session = self.db.get_session()
        positions = session.query(Position).filter(Position.is_active == True).all()
        session.close()
        return positions
