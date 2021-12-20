import gi
import sys

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from loguru import logger

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.success("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.opt(colors=True).warning(f"WARNING: {err}")
        logger.debug(f"DEBUG: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"ERROR: {err}")
        logger.debug(f"DEBUG: {debug}")
        loop.quit()
    return True
