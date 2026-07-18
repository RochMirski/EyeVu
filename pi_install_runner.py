import paramiko
import sys
import time

def ssh_execute(host, username, password, commands, timeout=3600):
    """Execute commands on remote host via SSH."""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print(f"Connecting to {username}@{host}...")
        client.connect(host, username=username, password=password, timeout=30)
        print("✓ Connected")
        
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            
            # Stream output in real-time
            while True:
                line = stdout.readline()
                if not line:
                    break
                print(line.rstrip())
            
            # Check for errors
            stderr_output = stderr.read().decode()
            if stderr_output:
                print("STDERR:", stderr_output)
            
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                print(f"Command failed with exit code {exit_code}")
                return False
        
        client.close()
        return True
        
    except Exception as e:
        print(f"SSH Error: {e}")
        return False

if __name__ == "__main__":
    host = "192.168.137.133"
    username = "roch"
    password = "1111"
    
    # Commands to execute
    commands = [
        "bash /home/roch/install_pi_ncnn.sh"
    ]
    
    success = ssh_execute(host, username, password, commands)
    sys.exit(0 if success else 1)
