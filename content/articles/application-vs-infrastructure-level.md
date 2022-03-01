---
title: "Application vs infrastructure level"
date: 2022-03-01
draft: false
type: "article"
---

_TL;DR: Choosing at which level between application or infrastructure you will act is an early and fundamental decision for your product, it needs to be carefully decided because it will impact the business broadly._

## SaaS

Software as a Service is the approach that lets you **build the way you want** without worrying about the support of multi-platforms.

Technical compatibility appears in the integration part but not on the design decisions made inside the product. You can choose your language, stack, structure, practices, to make your service run.

In short: you can **focus on the business needs**, not on the requirements due to the technical debt of your clients.

## Serve or Distribute

Software as a Service implies to serve your service, it can then be called via a network and/or the Internet. Certain services are accessible through a web interface others will have an API to access it. In this pattern, the service needs to be online.

But there is a different approach to provide your software: **bundling and distributing it**. Like the current application model on your phone and the famous AppStore. Your application is prepared, bundled, and sent to the store where people can download it and use it directly on their phones. This model allows the offline mode. Some challenges will hit when an update has to be done. Now it only relies on the user if they did or not the update. You can **combine the two models to enable your full product potential.**

## Architecture

The way you provide your software can have an impact on the overall architecture. If you choose to have an online service where users can connect to it and use your product directly, you can design your service as one product that can **support multiple users**, or a product that can **support one user** but you start multiple instances of your product to serve more clients.

A simple representation of the architecture:

_Multi-tenancy, multi-client product:_
```
- Product
	- Client 1
		- User 1
		- User n
	- Client 2
		- User 1
		- User n
	- Client n
		- ...
```

*_Multi-instance, single client product:_*

```
- Product: Instance for client 1
	- User 1
	- User n
- Product: Instance for client 2
	- User 1
	- User n
- Product: Instance for client n
	...
```

These two models of architecture imply a lot of differences under the hood and come with their advantages and constraints. It has an impact at a different level.

The first approach manages the multi-tenant at an application level, the second one handles the multi-tenant at an infrastructure level.

This has an impact at different levels of the stack:

_Levels:_
```
+-------------+
|   Product   | <- Application level
|-------------|
|  Instance   | <- Infrastructure level
+-------------+
```

If you want to handle multi-tenants at an application level, you have to build the **permissions system into the application itself**, define what a user can do and access. The limit and boundaries live inside the application.

At an infrastructure level, the application is not touched and the boundaries between users are made into the infrastructure configuration, like a new server or networking settings.

Acting at a different level has an impact on the business itself, the resources allocated will be different, and the teams will require a different skillset. At an application level, Software Engineers are more involved, and at an infrastructure level, DevOps will be the main actors.

As well, designing your product at an infrastructure level can allow flexibility in terms of distribution. Your product can be instanced in another place on another server. But with this approach, some constraints exist. The communication between clients will be more complex. It can be an advantage to keep your client data separated but features that federate data and allow interaction can be more complex to build.

Regarding data storage, at an application level, you can create one central database where all the tenants' data is stored. The separation can be made into the database with different users, tables and schemas.

At an infrastructure level, you have to deploy a new database per client and manage this database independently from the other DBs. So the data is well separated but it will not allow you to do cross tenant queries. If you want to aggregate data from every client you have to run a query on each DB. Also, sharing common data requires a different approach: instead of using one table and allowing all the users to query it, the table needs to be available in another common DB or duplicated into each DB.

### Deployment of a new version

When you work at an application level, all the clients can get access to the latest version of your product as soon as you release it.

At an infrastructure level, releasing a new version means deploying a new version for each client independently.

This question between application level and infrastructure level is becoming more obvious when you have to work on-premise. Deploying an application on the client infrastructure directly brings new challenges and if you have built a multi-tenant application it will not make sense to deploy that application on the client infrastructure. Indeed, you want to distribute only one instance of the application to the client and not a system to manage multiple instances of the application.

With that in mind, this kind of distribution will impact the design of the application.

Designing an application to be able to run either as SaaS or on-premise requires a different architecture and comes with new challenges to solve.

If you want to distribute the **same product to multiple clients**, the challenges you have to solve are:

- **Version management:** if you update the software, how can you distribute the updates across the different instances?
- **Monitoring:** how do you monitor multiple applications and aggregate the status of the service?
- **Resources:** each instance requires its own resources and has to be managed independently.
- **Infrastructure constraints:** where you deploy our application can also have an impact on its design. Indeed, if you deploy on a certain cloud provider some features/tools may be implemented differently or may not be accessible at all.

Standardization can avoid complex management. Having a replica instead of a custom version of your product per client will help.

“Cloud lock” can be mitigated by containers, open-source, and avoiding cloud-native services.

## Distribution isn’t integration

After having built the software you want to distribute. Above we described two different ways to distribute it: SaaS or Bundle.

Now how do you make your software **interact with the client system**? This is the role of **integration**. It does not impact your internal system but only the I/O of it. You can create many integrations to enable different use cases.

**e.g.:** API REST, S3 connector, Kafka, or even ipc, etc

The integration part describes how your software will connect to the rest of the system. An integration can be different based on the way you choose to distribute the software. This step deals with the design and the architecture.

## Impacts

The impacts of software distribution on the architecture:

### Cost

By building a service on your infrastructure, you have to manage the cost: cutting the bills becomes a real thing, resources usage is a concern. So you can combine services, use shared caching, ... This kind of cost reduction can be made with less friction when you manage the infrastructure.

### Move fast

Delegating tasks and adding vendors in the loop decrease the flexibility you have and create bottlenecks that can lead to slow iterations. If your product is deployed on-premise you will have to deal with the tech team of your client to make any fix or update.

### Security

Security becomes a thing when you deploy on-premise, as you rely on the security of your customer infrastructure. But your tool can also bring security holes in their system. Miss-configuration due to a lack of knowledge of the targeted system becomes a real weakness. You have to take care of the security of the software you built but the infrastructure security will rely mostly on the customer.

### Intellectual Property

By the time you distribute the binary or the source code, you can’t protect yourself anymore against someone stealing your work. Only licenses, contracts, and laws can protect you against that. But the technical barriers are almost gone.

Distributing your service through an API allows you to manage who can access it and they are only allowed to execute it, the intellectual property can be more protected with that form of distribution.

## In the wild

If you look at [Sentry.io](http://sentry.io) this tool is a perfect example, they chose to entirely [open source](https://github.com/getsentry/sentry) the product and they also provide Sentry as a SaaS. The product manages only one organization. To manage multiple organizations as they do for their SaaS, they have built other tools to manage multiple instances.

GitLab has the same pattern, they have an open-source version and they have a cloud service you can directly subscribe to, and they manage the instance for you. The open-source version is only for one organization; if you want to have **multiple organizations you need to start new instances**. So this is handled at an infrastructure level.

## So what

_Multi tenancy:_
```
+--------------------------------+
| Tenant 1 | Tenant 2 | Tenant 3 |
+--------------------------------+
|              App               |
+--------------------------------+
```
or _Single tenant:_
```
+--------------------------------+
| Tenant 1 | Tenant 2 | Tenant 3 |
+----------|----------|----------+
|   App    |   App    |   App    |
+--------------------------------+
```

Build at an application level or at an infrastructure level requires different skills and has an impact on the team resource. This choice will have an effect down the line and need to be aligned with the business objectives.
\
\
\
[Discuss on Twitter](https://twitter.com/YanaelBarbier)
