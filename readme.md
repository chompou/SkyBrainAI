# Skybrain-AI 🧠🤖
AI which aims to solve the [SkyBlock](https://skyblock.net) Minecraft survival challange. The network is created by using the NEAT algorithm.
![image](https://user-images.githubusercontent.com/7690439/200194574-91f809b6-131b-417a-9d28-652a5fb69669.png)
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
pip install minedojo, stable-baselines3, torchvision
```
