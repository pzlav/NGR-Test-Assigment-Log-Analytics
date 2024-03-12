# NGRSoftlab test assigment (Log Message Analytics)
### Description
This project is designed as a test assignment for candidates. It involves starting a script named generate.py, which loads a language model trained on a significant log file and begins generating messages to port 10514. The core tasks include collecting these messages, parsing them, storing them in a local database (preferably ClickHouse), and conducting various analytics on the collected data.

## Getting Started
### Dependencies
Python 3.x
ClickHouse or any preferred local database system capable of handling large datasets
Install requirents txt file


### Executing the Project
Start the generate.py script to begin log message generation:
Collect messages from local port 10514. 

# Implement your parsing and storage logic here
- Parse the collected messages. This step will require creativity, as message formats can vary widely. You may need to write custom parsing functions based on observed patterns in the data.

- Store the parsed messages in ClickHouse (or your chosen database). Ensure your database schema is designed to optimally store and query the data.

- Perform analytics on your data. This can include clustering, visualization, anomaly detection, or any other statistical analysis that provides insights into the data. Python's pandas, matplotlib, and seaborn libraries will be useful for this step.


## Contributing
Contributions to this project are welcome. Please follow these steps to contribute:


## License
This project is licensed under the MIT License