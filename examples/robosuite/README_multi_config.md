# Multi-Configuration Hanoi Implementation

This directory contains an enhanced version of the Hanoi environment that can handle multiple block configurations, similar to the dataset making module.

## Files

- `main_hanoi_multi_config.py` - Main multi-configuration Hanoi script
- `test_multi_config.py` - Test script to verify functionality
- `main_hanoi.py` - Original single-configuration script (for reference)

## Key Features

### Multi-Configuration Support
- **Random Block Placement**: Places blocks on pegs randomly according to Towers of Hanoi rules
- **Random Block Selection**: Randomly selects 3 out of 4 available blocks
- **Dynamic Plan Generation**: Uses the PDDL planner to generate plans for different configurations
- **Adaptive Task Sequences**: Creates task sequences based on the generated plan

### Integration with Dataset Making Module
- Uses `RecordDemos` class from the dataset making module
- Integrates with the PDDL planning system
- Supports the same detector system (`PandaHanoiDetector`)
- Maintains compatibility with the original task sequence management

### Enhanced Task Management
- Dynamic task generation based on the current block configuration
- Automatic task completion checking using both detectors
- Support for both sequential and single-prompt modes
- Timeout handling for individual tasks

## Usage

### Basic Usage
```bash
python main_hanoi_multi_config.py --episodes 10 --random-block-placement --random-block-selection
```

### Configuration Options
- `--random-block-placement`: Enable random block placement on pegs
- `--random-block-selection`: Enable random selection of 3 out of 4 blocks
- `--cube-init-pos-noise-std`: Standard deviation for initial position noise
- `--use-sequential-tasks`: Use sequential task prompts (default: True)
- `--time-based-progression`: Advance tasks based on time rather than completion
- `--task-timeout`: Timeout for individual tasks in steps

### Testing
```bash
python test_multi_config.py
```

## Architecture

### MultiConfigHanoiEnvironment
The main environment class that:
- Sets up the Robosuite environment with multi-config support
- Integrates with the dataset making module's `RecordDemos` class
- Generates dynamic task sequences based on the current configuration
- Manages both simple and ground truth detectors

### TaskManager
Handles task progression and completion:
- Tracks task completion using detector predicates
- Supports both sequential and single-prompt modes
- Implements timeout handling
- Provides completion statistics

### Dynamic Plan Generation
The system automatically:
1. Detects the current block configuration using `PandaHanoiDetector`
2. Generates a PDDL problem file based on the current state
3. Calls the planner to generate a sequence of actions
4. Converts the PDDL plan to natural language task descriptions
5. Creates completion check functions for each task

## Key Differences from Original

### Original `main_hanoi.py`
- Fixed task sequence for 3 blocks
- Single configuration support
- Manual task definition
- No planning integration

### New `main_hanoi_multi_config.py`
- Dynamic task generation based on current configuration
- Support for 3-4 blocks with random selection
- Integrated PDDL planning system
- Automatic plan-to-task conversion
- Full compatibility with dataset making module

## Dependencies

- Robosuite
- OpenPI client (for policy inference)
- Dataset making module components
- Planning module (PDDL)
- Wandb (for logging)
- ImageIO (for video recording)

## Configuration Examples

### Random Configuration
```bash
python main_hanoi_multi_config.py \
    --random-block-placement \
    --random-block-selection \
    --episodes 50 \
    --use-sequential-tasks
```

### Fixed Configuration
```bash
python main_hanoi_multi_config.py \
    --no-random-block-placement \
    --no-random-block-selection \
    --episodes 20
```

### Time-based Progression
```bash
python main_hanoi_multi_config.py \
    --time-based-progression \
    --task-timeout 200 \
    --episodes 30
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the dataset making module is in the Python path
2. **Planning Failures**: Check that the PDDL files are accessible
3. **Detector Issues**: Verify that the environment has the required objects
4. **Memory Issues**: Reduce episode count or horizon for testing

### Debug Mode
Enable debug logging to see detailed information:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Support for more complex block configurations
- Integration with additional planning algorithms
- Enhanced visualization of task progression
- Support for custom goal states
- Integration with additional robot platforms
