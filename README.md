# VLSI Design

VLSI (Very Large Scale Integration) refers to the trend of integrating circuits into silicon chips.
A typical example is the smartphone. The modern trend of shrinking transistor sizes, allowing engineers to 
fit more and more transistors into the same area of silicon, has pushed the integration of more and more functions 
of cellphone circuitry into a single silicon die (i.e. plate). This enabled the modern cellphone to mature into a 
powerful tool that shrank from the size of a large brick-sized unit to a device small enough to comfortably carry 
in a pocket or purse, with a video camera, touchscreen, and other advanced features.

## Usage

The project makes use of a Docker container. To execute it, first you have to build the image, by specifying a name
for it using the flag `-t`, for example, `cdmo-vlsi`:

```commandline
docker build . -t <image-name>
```

The `.` makes sure docker looks for `Dockerfile` in the current folder.

Then, we have to start the container, by using volumes to bind the project folder on the local machine to the working
directory on the container:

```commandline
docker run -v <local-path>:<docker-container-path>
```