from openenv import SyncEnvClient
from env import EmailEnv

# Connect your environment to OpenEnv
client = SyncEnvClient(env_class=EmailEnv)