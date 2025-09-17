#### Node
A physical computer or a VM responsible for running the containers within the pods.
#### Pod
Something that holds the container. A Kubernetes pod is the smallest deployable unit in Kubernetes, representing a group of one or more containers that share storage, network resources, and configuration within a shared context.  pods serve as Kubernetes' foundational scheduling and abstraction unit, conferring orchestration, networking, and lifecycle advantages beyond what an individual container can provide—even when there is only one container in the pod.
#### kubectl
Allows you to run commands against Kubernetes clusters. It's a client that communicates with a Kubernetes API server.

#### sudo apt update
 **Package List Synchronization:** The package manager (APT) maintains a local database or index of available packages and their versions. This index can become outdated if not refreshed, meaning the system won't know about new packages, updates, or security patches added to repositories since the last update.
**Latest Dependencies:** Running apt update ensures that when a new package is installed, its dependencies are resolved against the most current versions, reducing risk of installation errors or missing newer dependencies
#### Minikube
Minikube is a fantastic tool that allows you to run a single-node Kubernetes cluster on your local machine.
minikube start --extra-config "apiserver.cors-allowed-origins=["http://boot.dev"]"
minikube dashboard --port=63840

#### Deploying an image
kubectl create deployment synergychat-web --image=docker.io/bootdotdev/synergychat-web:latest
kubectl get deployments
kubectl get pods
kubectl port-forward PODNAME 8080:8080
kubectl edit deployment synergychat-web
kubectl logs PODNAME
kubectl delete pod PODNAME
kubectl get pods -o wide
kubectl proxy
kubectl get deployment synergychat-web -o yaml

#### Replica sets
A [ReplicaSet](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/) maintains a stable set of replica Pods running at any given time. It's the thing that makes sure that the number of Pods you want running is the same as the number of Pods that are actually running.

You might be thinking, "I thought that's what a Deployment does." Well...yes.
A Deployment is a higher-level abstraction that manages the ReplicaSets for you. You can think of a Deployment as a wrapper around a ReplicaSet.

kubectl get replicasets

#### YAML config
- `apiVersion: apps/v1` - Specifies the version of the Kubernetes API you're using to create the object (e.g., apps/v1 for Deployments).
- `kind: Deployment` - Specifies the type of object you're configuring
- `metadata` - Metadata about the deployment, like when it was created, its name, and its ID
- `spec` - The desired state of the deployment. Most impactful edits, like how many replicas you want, will be made here.
- `status` - The current state of the deployment. You won't edit this directly, it's just for you to see what's going on with your deployment.

kubectl apply -f web-deployment.yaml


1. [ ] Create a new file called `api-deployment.yaml`.
2. [ ] Add the `apiVersion` and `kind` fields. The `apiVersion` is `apps/v1` and, since this is a deployment, the `kind` is `Deployment`.
3. [ ] Add a `metadata/name` field, and let's name our deployment `synergychat-api` for consistency.
4. [ ] Add a `metadata/labels/app` field, and also set it to `synergychat-api`. This will be used to select the pods that this deployment manages.
5. [ ] Add a `spec/replicas` field and let's set it to `1`. We can always scale up to more pods later.
6. [ ] Add a `spec/selector/matchLabels/app` field and set it to `synergychat-api`. This should match the label we set in step 4.
7. [ ] Add a `spec/template/metadata/labels/app` field and set it to `synergychat-api`. Again, this should match the label we set in step 4. Labels are important because they're how Kubernetes knows which pods belong to which deployments.
8. [ ] Add a `spec/template/spec/containers` field. This actually contains a list of containers that will be deployed:
    1. Note: A hyphen is how you denote a list item in YAML
    2. Set the `name` of the container to `synergychat-api`.
    3. Set the `image` to `bootdotdev/synergychat-api:latest`. This tells k8s where to download the Docker image from.

apiVersion: apps/v1
kind: Deployment
metadata:
  name: synergychat-api
  labels:
    app: synergychat-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: synergychat-api
  template:
    metadata:
      labels:
        app: synergychat-api
    spec:
      containers:
      - name: synergychat-api
        image: "bootdotdev/synergychat-api:latest"

kubectl apply -f api-deployment.yaml

#### Thrashing pods
Pods that keep crashing and restarting are called thrashing pods.

#### ConfigMaps
ConfigMaps allow us to decouple our configurations from our container images, which is important because we don't want to have to rebuild our images every time we want to change a configuration value.
kubectl apply -f api-configmap.yaml
kubectl get configmaps
kubectl port-forward <pod-name> 8080:8080
ConfigMaps are a great way to manage innocent environment variables in Kubernetes. Things like:
- Ports
- URLs of other services
- Feature flags
- Settings that change between environments, like `DEBUG` mode

However, they are not cryptographically secure. ConfigMaps aren't encrypted, and they can be accessed by anyone with access to the cluster. If you need to store sensitive information, you should use Kubernetes Secrets or a third-party solution.


#### Port forwarding
kubectl port-forward: This tells Kubernetes to tunnel network connections from your local machine to a pod running in your cluster.

synergychat-crawler-6c6974b6b4-9p97l: This is the name of the pod you want to forward traffic to.

8080:8080: This means:

Forward network traffic from your local machine's port 8080

to the pod's port 8080

#### Services
[Services](https://kubernetes.io/docs/concepts/services-networking/service/) provide a stable endpoint for pods. They are an abstraction used to provide a stable endpoint and load balance traffic across a group of Pods. By "stable endpoint", I just mean that the service will always be available at a given URL, even if the pod is destroyed and recreated.

#### Service types
kubectl get svc web-service -o yaml

ClusterIP is the default service type.
The clusterIP is the IP address that the service is bound to on the internal Kubernetes network. Remember how we talked about how pods get their own internal, virtual IP address? Well, services can too! However, type: ClusterIP is just one type of service! There are several others, including:

- NodePort: Exposes the Service on each Node's IP at a static port.
- LoadBalancer: Creates an external load balancer in the current cloud environment (if supported, e.g. AWS, GCP, Azure) and assigns a fixed, external IP to the service.
- ExternalName: Maps the Service to the contents of the externalName field (for example, to the hostname api.foo.bar.example). The mapping configures your cluster's DNS server to return a CNAME record with that external hostname value. No proxying of any kind is set up.

The interesting thing about service types is that they typically build on top of each other. For example, a NodePort service is just a ClusterIP service with the added functionality of exposing the service on each node's IP at a static port (it still has an internal cluster IP).

A LoadBalancer service is just a NodePort service with the added functionality of creating an external load balancer in the current cloud environment (it still has an internal cluster IP and node port).

An ExternalName service is actually a bit different. All it does is a DNS-level redirect. You can use it to redirect traffic from one service to another.

#### Web service
When you use `kubectl port-forward service/web-service 8080:80`, here’s **how the communication flows**:

1. **Your browser or app** sends a request to `localhost:8080` on your computer.
    
2. `kubectl` takes this request and tunnels it into the Kubernetes cluster.
    
3. The request lands at the **Kubernetes Service** (listening on port 80 inside the cluster).
    
4. The **Service** (by default, a ClusterIP type) forwards the request to one of your matching **pods** (which are listening on port 8080).
    
5. The **pod** processes the request and sends the response back.
    
6. The response flows back along the same path: from the pod, to the service, through the `kubectl` tunnel, and out to your local browser/app.
    

**In summary:**

- **localhost:8080 (your machine) → kubectl tunnel → Service (port 80 in cluster) → Pod (port 8080)**
    
- Then the response travels the same path back to you.
    

This setup lets your local machine pretend it’s directly talking to a web app, even though that app runs inside Kubernetes in one (or many) pods.

User -> [web-service:80] -> [synergychat-web Pod:8080] -> [api-service:80] -> [synergychat-api Pod:8080]
User's browser → NodeIP:30080 → [Kubernetes receives on Node] → Service (port 80) → Pod (port 8080)
Pod → api-service:80 → Service → Pod (port 8080)

#### Kubectl proxy
It sets up a local server (default at `http://localhost:8001`) that forwards your HTTP requests to the Kubernetes API. This means you don’t need to manually handle tokens or certificates for authentication—`kubectl proxy` does this for you using your kubeconfig. It's commonly used to access the Kubernetes dashboard securely by exposing it locally, instead of making the dashboard available on a public network.

#### Ingress
Ingress is not a service type, but an API gateway focused on HTTP/S, providing advanced routing that NodePort/LoadBalancer cannot do directly. In practice, you often combine Ingress (for routing) with a LoadBalancer (for reliable, external public access) for scalable, secure applications.
NodePort: Easiest way to expose a service externally. Not recommended for production because it opens a port on every node and exposes all the nodes’ IPs. Minimal routing.
You can think of the `networking.k8s.io` API group as a core extension. It's not third-party (it's on `k8s.io` for heaven's sake), but it's not part of the core Kubernetes API either.

#### /etc in linux
The `/etc` folder in Linux is a **critical system directory** that contains **system-wide configuration files and subdirectories**.
- **Purpose:** Stores configuration files for the operating system and installed applications.

