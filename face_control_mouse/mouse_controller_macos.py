"""
This module contains functions for controlling the mouse on macOS using the
Quartz CoreGraphics library.
"""
from Quartz.CoreGraphics import (
    CGEventCreateMouseEvent,
    CGEventPost,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
    kCGHIDEventTap
)

def _mouse_event(event_type, pos_x, pos_y, mouse_button):
    """Helper function to create and post a mouse event."""
    event = CGEventCreateMouseEvent(None, event_type, (pos_x, pos_y), mouse_button)
    CGEventPost(kCGHIDEventTap, event)

def move(pos_x, pos_y):
    """Moves the mouse to the specified coordinates."""
    _mouse_event(kCGEventMouseMoved, pos_x, pos_y, kCGMouseButtonLeft) # Button doesn't matter for move

def left_click(pos_x, pos_y):
    """Performs a left click at the specified coordinates."""
    _mouse_event(kCGEventLeftMouseDown, pos_x, pos_y, kCGMouseButtonLeft)
    _mouse_event(kCGEventLeftMouseUp, pos_x, pos_y, kCGMouseButtonLeft)

def right_click(pos_x, pos_y):
    """Performs a right click at the specified coordinates."""
    _mouse_event(kCGEventRightMouseDown, pos_x, pos_y, kCGMouseButtonRight)
    _mouse_event(kCGEventRightMouseUp, pos_x, pos_y, kCGMouseButtonRight)