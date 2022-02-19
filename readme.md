# BigSpark

A project for the course *2AMD15 - Big Data Management*

## Development environment

A live JupyterLab environment for this project is hosted on Azure and available at [http://52.186.66.122/](http://52.186.66.122/).

The lab is deployed using the `jupyter/all-spark-notebook` Docker container with support for Spark/PySpark. For more information on which packages are available, see [here](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-all-spark-notebook).

The `--collaborative` feature is enabled for this JupyterLab, meaning multiple people can access and edit the notebooks at the same time by simply following the link above and provide the correct credentials.

:exclamation: **NOTE:** The lab is running on the free **100$** credits available in the [Azure for Students](https://azure.microsoft.com/en-us/free/students/) subscription. Even though this is free, Azure VMs are expensive af, to prevent running out of credits mid-project, I set up the lab to auto shutdown if it is idle for more than 30 minutes. Therefore, if you cannot access the lab from the link above, it means the server is shut down, don't be pannic and follow the instructions [below](#manual-startstop-jupyterlab-server) to start it again.

### Manual start/stop JupyterLab

To **start** the lab, make an empty `POST` request to the following webhook `https://a5729cdd-7963-4cc3-9063-02f784998958.webhook.eus.azure-automation.net/webhooks?token=07Q1%2fBUaLVx5IMH3M1upIxajK5xrlihpFTGw1s4mNLM%3d`, or simply run the following command in your CLI:

    curl -X POST -d "" https://a5729cdd-7963-4cc3-9063-02f784998958.webhook.eus.azure-automation.net/webhooks?token=07Q1%2fBUaLVx5IMH3M1upIxajK5xrlihpFTGw1s4mNLM%3d

It usually takes 5-10 minutes for the lab to be accessible.

To **stop** the lab (you don't need to do this manually because of the auto shutdown), make an empty `POST` request to the following webhook `https://a5729cdd-7963-4cc3-9063-02f784998958.webhook.eus.azure-automation.net/webhooks?token=uoMjc%2fbEaX4Ty05GOCFByIvqlK7MJM%2fDJ0NgQ51qS3c%3d`, or simply run the following command in your CLI:

    curl -X POST -d "" https://a5729cdd-7963-4cc3-9063-02f784998958.webhook.eus.azure-automation.net/webhooks?token=uoMjc%2fbEaX4Ty05GOCFByIvqlK7MJM%2fDJ0NgQ51qS3c%3d