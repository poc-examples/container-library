# Use the official Apache HTTP Server image
FROM docker.io/httpd:latest

# Change the listen port of Apache to 8080
RUN sed -i 's/Listen 80/Listen 8080/' /usr/local/apache2/conf/httpd.conf

# Copy your HTML file to the directory served by Apache
COPY html/index.html /usr/local/apache2/htdocs/

RUN chgrp -R 0 /usr/local/apache2/htdocs/ /usr/local/apache2/logs && \
    chmod -R g=u /usr/local/apache2/htdocs/ /usr/local/apache2/logs

USER 1001
