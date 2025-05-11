import sys
import argparse
import socket
import driver
import time

# Configure argument parser
parser = argparse.ArgumentParser(description='Python client for TORCS SCRC server.')
parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Max learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Max steps per episode (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Track name')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

arguments = parser.parse_args()

# Print connection summary
print(f'Connecting to {arguments.host_ip}:{arguments.host_port}')
print(f'Bot ID: {arguments.id}, Max Episodes: {arguments.max_episodes}, Max Steps: {arguments.max_steps}')
print('*')

# Create UDP socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)  # Set timeout for socket operations
except socket.error as e:
    print(f'Socket creation failed: {e}')
    sys.exit(-1)

shutdownClient = False
d = driver.Driver(arguments.stage)

# Connection establishment with retry logic
max_retries = 5
retry_count = 0
connected = False

while not connected and retry_count < max_retries:
    try:
        print(f'Sending ID: {arguments.id}')
        init_msg = arguments.id + d.init()
        sock.sendto(init_msg.encode(), (arguments.host_ip, arguments.host_port))
        
        while True:
            try:
                buf, addr = sock.recvfrom(1000)
                response = buf.decode()
                print(f"Received response: {response}")
                
                if 'identified' in response:
                    print('Connection established')
                    connected = True
                    break
                elif 'shutdown' in response:
                    print('Server requested shutdown')
                    d.onShutDown()
                    shutdownClient = True
                    break
            except socket.timeout:
                print('No response received, retrying...')
                break
                
    except socket.error as e:
        print(f'Connection error: {e}')
        retry_count += 1
        if retry_count < max_retries:
            print(f'Retrying... ({retry_count}/{max_retries})')
            time.sleep(1)
        else:
            print('Max retries reached. Exiting...')
            sys.exit(-1)

if not connected:
    print('Failed to establish connection')
    sys.exit(-1)

# Main loop
print("Entering main loop")
while not shutdownClient:
    try:
        # Receive message from server
        buf, addr = sock.recvfrom(1000)
        msg = buf.decode()
        
        # Handle special messages
        if 'shutdown' in msg:
            print('Received shutdown command')
            d.onShutDown()
            shutdownClient = True
            break
        elif 'restart' in msg:
            print('Received restart command')
            d.onRestart()
            continue
            
        # Process the message and get action
        action = d.drive(msg)
        if not action:
            print('No action generated, skipping...')
            continue
            
        # Send action back to server
        try:
            sock.sendto(action.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as e:
            print(f'Error sending action: {e}')
            continue
            
    except socket.timeout:
        print('Timeout waiting for server message')
        continue
    except socket.error as e:
        print(f'Network error: {e}')
        continue
    except Exception as e:
        print(f'Unexpected error: {e}')
        continue

# Cleanup
try:
    sock.close()
    print("Client terminated")
except:
    pass