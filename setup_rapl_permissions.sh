#!/bin/bash
# Script to set up RAPL energy file permissions

# Option 1: Create a udev rule (recommended, persistent after reboot)
# This will make RAPL files readable by all users
sudo tee /etc/udev/rules.d/99-rapl.rules > /dev/null << 'EOF'
# Allow users to read RAPL energy files
SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod 444 %S/subsystem/%p/energy_uj"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=powercap

echo "Udev rule created. You may need to unplug/replug or reboot for changes to take effect."

# Option 2: Manually change permissions (temporary, resets on reboot)
# Uncomment the line below to manually change permissions:
# sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj

