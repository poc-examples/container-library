package com.example;

import org.springframework.batch.core.JobExecution;
import org.springframework.batch.core.JobExecutionListener;

public class JobCompletionNotificationListener implements JobExecutionListener {

    @Override
    public void beforeJob(JobExecution jobExecution) {
        System.out.println("Job starting...");
    }

    @Override
    public void afterJob(JobExecution jobExecution) {
        System.out.println("Job finished with status: " + jobExecution.getStatus());
    }
}
