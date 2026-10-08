class RobotHostError(RuntimeError): pass
class AIServiceError(RobotHostError): pass
class DetectionError(RobotHostError): pass
class AmbiguousTargetError(DetectionError): pass
class GeometryError(RobotHostError): pass
class ConfigurationError(RobotHostError): pass
