from pydantic import ValidationError


class RagError(Exception):
    """
    Base class for all domain-specific errors in the system.

    Attributes:
        default_message: Fallback message used when no message is provided.
    """

    default_message = "An unspecified error occurred."

    def __init__(self, message: str | None = None) -> None:
        """
        Initializes the exception with a custom or default message.

        Args:
            message: Specific error details or None to use the default message.
        """
        super().__init__(message or self.default_message)


class SchemaValidationError(RagError):
    """Error raised when data fails Pydantic validation checks."""

    def __init__(
        self,
        e: ValidationError,
        context: str | None = None,
    ) -> None:
        """
        Initializes the validation error with formatted messages.

        Args:
            e: The validation error object.
            context: Information about where the error occurred.
        """
        header = (
            f"Validation failed in '{context}':"
            if context
            else "Validation failed:"
        )
        messages: list[str] = [header]

        for error in e.errors():
            location = (
                " -> ".join(str(loc) for loc in error["loc"])
                if error["loc"]
                else "model"
            )
            err_value = (
                f" invalid value: <{error['input']}> -"
                if "input" in error and isinstance(error["input"], str)
                else ""
            )
            messages.append(f"- ({location}){err_value} {error['msg']}")

        super().__init__("\n".join(messages))
