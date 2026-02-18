#!/usr/bin/env python3
import pty
import os
import sys
import select
import termios
import tty

def ssh_with_pty():
    """Run SSH with proper PTY that can accept input"""
    cmd = ['ssh', 'nibi', 'exit']

    pid, fd = pty.fork()

    if pid == 0:
        # Child process
        os.execvp(cmd[0], cmd)
    else:
        # Parent process
        try:
            # Set stdin to raw mode to pass through input
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

            while True:
                # Use select to monitor both SSH output and stdin
                r, w, e = select.select([sys.stdin, fd], [], [], 0.1)

                # If SSH has output, print it
                if fd in r:
                    try:
                        data = os.read(fd, 1024)
                        if data:
                            os.write(sys.stdout.fileno(), data)
                        else:
                            break
                    except OSError:
                        break

                # If user has input, send it to SSH
                if sys.stdin in r:
                    try:
                        data = os.read(sys.stdin.fileno(), 1024)
                        if data:
                            os.write(fd, data)
                    except OSError:
                        break

        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            os.waitpid(pid, 0)
            print("\n[SSH session ended]")

if __name__ == "__main__":
    ssh_with_pty()
