class DartApiError(Exception):
    """DART Open API가 성공이 아닌 상태 코드를 반환했을 때 발생시킨다."""

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(f"[{status}] {message}")
