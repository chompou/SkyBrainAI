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
### 2. Installing MineRL
Start by running the following command.
```bash
pip install --verbose git+https://github.com/minerllabs/minerl
```

### 3. Configure MineRL settings
Navigate and open the file `./venv/lib/minerl/MCP../initiateRun`
Then change the line containing
```
java -Xmx$maxMem -jar $fatjar --envPort=$port
```
with
```
java -Xmx$maxMem -XstartOnFirstThread -jar $fatjar --envPort=$port
```

