"""
Security middleware and utilities for Flask application
Handles authentication, rate limiting, and security headers
"""
from functools import wraps
from flask import current_app, g, jsonify, request
from typing import Callable, Any
import hmac
import logging
import os

logger = logging.getLogger(__name__)


SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _auth_failure(message: str, status_code: int = 401):
    """Return a consistent JSON response for protected API routes."""
    error = "Unauthorized" if status_code == 401 else "Forbidden" if status_code == 403 else "Service Unavailable"
    return jsonify({"error": error, "message": message}), status_code


def _current_auth_manager():
    manager = current_app.extensions.get("auth_manager")
    if manager is None:
        raise RuntimeError("Authentication manager has not been initialized")
    return manager


def _authenticate_request(require_csrf: bool = True):
    """Resolve a request session once and put its safe user on Flask ``g``."""
    from auth_manager import AuthenticationUnavailableError

    manager = _current_auth_manager()
    if not manager.auth_enabled():
        # Explicit development escape hatch only.  It is deliberately off by
        # default and must never be used in production.
        return None
    try:
        user, session = manager.get_session_user(request.cookies.get(current_app.config["AUTH_SESSION_COOKIE"]))
    except AuthenticationUnavailableError:
        return _auth_failure("Authentication service is unavailable", 503)
    if not user or not session:
        return _auth_failure("Log in to continue")
    if require_csrf and request.method not in SAFE_METHODS:
        csrf_cookie = request.cookies.get(current_app.config["AUTH_CSRF_COOKIE"])
        csrf_header = request.headers.get("X-CSRF-Token")
        if not manager.validate_csrf(session, csrf_cookie, csrf_header):
            return _auth_failure("Invalid security token", 403)
    g.current_user = user
    g.auth_session = session
    return None


def require_login(f: Callable) -> Callable:
    """Require a valid MongoDB browser session for one Flask route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        failure = _authenticate_request()
        if failure is not None:
            return failure
        return f(*args, **kwargs)
    return decorated_function


def require_roles(*roles: str) -> Callable:
    """Require a login and at least one of the supplied access roles."""
    allowed_roles = set(roles)

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            failure = _authenticate_request()
            if failure is not None:
                return failure
        
            if not _current_auth_manager().auth_enabled():
                return f(*args, **kwargs)
            user_roles = g.current_user.get("roles", g.current_user.get("role"))
            if not _current_auth_manager().role_has_any(user_roles, allowed_roles):
                return _auth_failure("Your account does not have permission for this action", 403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_search_api_key(f: Callable) -> Callable:
    """Protect the external search endpoint with one server-configured key.

    The operator manually sets ``SEARCH_API_KEY`` in the backend ``.env``.
    External callers send the same value only through ``X-API-Key``. The key
    is never accepted in a URL, returned in a response, or written to logs.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        configured_key = os.environ.get("SEARCH_API_KEY", "").strip()
        supplied_key = request.headers.get("X-API-Key", "").strip()
        if len(configured_key) < 32:
            logger.error("External search API is disabled: SEARCH_API_KEY is not securely configured")
            return _auth_failure("API-key authentication is unavailable", 503)
        if not supplied_key or not hmac.compare_digest(supplied_key, configured_key):
            logger.warning("Invalid API key attempt from %s", request.remote_addr)
            return _auth_failure("Invalid or missing API key")

        return f(*args, **kwargs)
    return decorated_function


class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, app=None, config=None):
        self.app = app
        self.config = config
        if app is not None:
            self.init_app(app, config)
    
    def init_app(self, app, config):
        """Initialize security features for Flask app"""
        self.app = app
        self.config = config
        
        # Register error handlers
        self._register_error_handlers()
    
    def _register_error_handlers(self):
        """Register custom error handlers to prevent information disclosure"""
        
        @self.app.errorhandler(400)
        def bad_request(e):
            return jsonify({
                "error": "Bad Request",
                "message": "Invalid request format"
            }), 400
        
        @self.app.errorhandler(401)
        def unauthorized(e):
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }), 401
        
        @self.app.errorhandler(403)
        def forbidden(e):
            return jsonify({
                "error": "Forbidden",
                "message": "Access denied"
            }), 403
        
        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({
                "error": "Not Found",
                "message": "Resource not found"
            }), 404
        
        @self.app.errorhandler(429)
        def ratelimit_handler(e):
            return jsonify({
                "error": "Too Many Requests",
                "message": "Rate limit exceeded. Please try again later."
            }), 429
        
        @self.app.errorhandler(500)
        def internal_error(e):
            logger.error(f"Internal server error: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please contact support."
            }), 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            # Log the actual error
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }), 500


def validate_request_size(max_size_kb: int = 16) -> Callable:
    """Decorator to validate request payload size. Usage: @validate_request_size() for default 16KB,
    or @validate_request_size(max_size_kb=2048) for a larger limit on bulk routes."""
    max_bytes = max_size_kb * 1024

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_length and request.content_length > max_bytes:
                return jsonify({
                    "error": "Payload Too Large",
                    "message": f"Request exceeds maximum allowed size of {max_size_kb}KB"
                }), 413
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def sanitize_input(data: str, max_length: int = 500) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        data: Input string to sanitize
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    """
    if not data:
        return ""
    
    # Limit length
    data = data[:max_length]
    
    # Remove potentially dangerous characters
    # Keep alphanumeric, spaces, and basic punctuation
    import re
    data = re.sub(r'[<>{}\\]', '', data)
    
    return data.strip()
