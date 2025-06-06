Change directory to assignment1
# cd assignment1

Then run the following commands to build Docker image and run on port 5000
# docker build -t ghcr.io/amuzaeem/zk-mlops01/assignment1:latest .
# docker run -d -p 5000:5000 ghcr.io/amuzaeem/zk-mlops01/assignment1:latest

Access the endpoint from PORTS tab --> see forward addresses for port 5000 --> Use the link to access the Liver Disease Prediction system UI
