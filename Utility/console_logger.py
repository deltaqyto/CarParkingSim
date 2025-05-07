

class ConsoleLogger:
    def __init__(self, verbosity_level='info', levels=None, silent=False):
        self.levels = ['debug', 'info', 'warning', 'error', 'fatal', 'none'] if levels is None else levels
        self.verbosity_level = verbosity_level if not silent else 'none'
        if verbosity_level not in self.levels:
            raise ValueError(f"Verbosity {verbosity_level} not in possible options {self.levels}")
        self.verbosity_index = self.levels.index(self.verbosity_level)

    def debug(self, caller, message):
        if not self._is_sufficient_level('debug'):
            return
        print(f'{type(caller).__name__}: ' + message)

    def info(self, caller, message):
        if not self._is_sufficient_level('info'):
            return
        print(f'{type(caller).__name__}: ' + message)

    def get_verbosity_levels(self):
        return self.levels

    def _is_sufficient_level(self, verbosity):
        return self.levels.index(verbosity) >= self.verbosity_index
