import argparse

def gen_sh_file(num_agents, sweep_command, project):
    script_content = f"""#!/bin/bash

NUM_AGENTS={num_agents}
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent {sweep_command} > {project}_${{HOSTNAME}}_${{agent_num}}.log 2>&1 &
done
"""

    with open(f'{project}.sh', 'w') as file:
        file.write(script_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--command', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    args = parser.parse_args()

    gen_sh_file(args.num_agents, args.command, args.project)
