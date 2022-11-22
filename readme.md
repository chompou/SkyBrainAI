# Skybrain-AI 🧠🤖
AI which aims to solve the [SkyBlock](https://skyblock.net) Minecraft survival challange. The network is created by using the Double DQN algorithm from StableBaselines3 customized with Prioritized Replay.
<img alt="image" height="465" src="https://user-images.githubusercontent.com/7690439/200194574-91f809b6-131b-417a-9d28-652a5fb69669.png"/>
## Prerequisits
- Java 1.8

## Setup project
### 1. Configuring Java JDK
#### On mac:
This will set Java 1.8 to your system-default Java version

Insert the following line to the file `~/.zshrc`
```text
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
```

Verify that the java-version is correct by running `java -version`. The output should be `1.8.x.x`
### 2. Installing Dependencies:
```text
pip install minedojo, stable-baselines3, torchvision, tensorboard

change variables guiScale and renderDistance in venv/lib/python../site-packages/minedojo/sim/Malmo/Minecraft/run/options.txt to 1
```


### 3. Run traning
To run the training, use the Jupyter Notebook [CoolNotz.ipynb](./CoolNotz.ipynb)

#### View training progress
During training, the training-agent will log it's performance using tensorboard. A tensorboard-server will need to be booted up to view this data.
This may be done by entering the following code in the terminal
```
python3 -m tensorboard.main --logdir ./DQN_steve_tensorboard/
```