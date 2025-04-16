from sqlalchemy import Column, Integer, String, DateTime, Numeric, Boolean, JSON, ForeignKey, Index, func
from .database import Base

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    mint = Column(String, nullable=False)
    price = Column(Numeric, nullable=False)
    volume = Column(Numeric, nullable=False)
    signal_type = Column(String)  # ENTRY EXIT STOPLOSS
    created_at = Column(DateTime, server_default=func.now())
    run_id = Column(String, nullable=False)

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    mint = Column(String, nullable=False)
    entry_price = Column(Numeric, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    position_size = Column(Numeric, nullable=False)
    token_amount = Column(Numeric, nullable=False)
    exit_price = Column(Numeric)
    exit_time = Column(DateTime)
    is_active = Column(Boolean, nullable=False)
    entry_signal_id = Column(Integer, ForeignKey('signals.id'))
    exit_signal_id = Column(Integer, ForeignKey('signals.id'))
    created_at = Column(DateTime, server_default=func.now()) 
    run_id = Column(String, nullable=False)