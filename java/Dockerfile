# Step 1: Use a Red Hat UBI builder image
FROM registry.access.redhat.com/ubi8/openjdk-11:1.3 AS builder

USER root

# Set the working directory in the builder image
WORKDIR /app

# Copy the Maven configuration files first and download dependencies
COPY pom.xml .

# Downloads all of the dependencies required for the project
RUN mvn dependency:go-offline

# Copy the rest of the application
COPY src ./src

# Build the application
# Running as root to avoid permission issues during build
RUN mvn package -DskipTests

# Step 2: Use a Red Hat UBI lightweight base image
FROM registry.access.redhat.com/ubi8/openjdk-11-runtime:1.18-2

USER root

# Set the working directory
WORKDIR /app

# Create a non-root user and group (OpenShift requirement)
RUN chgrp -R 0 /app && \
    chmod -R g=u /app

# Copy the built application from the builder image
COPY --from=builder /app/target/*.jar app.jar

# RUN chown -R 1001 /app
USER 1001

# Define the command to run the application
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
