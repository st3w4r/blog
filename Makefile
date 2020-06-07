.PHONY: build
build:
	docker-compose up build

.PHONY: server
server:
	docker-compose up server

.PHONY: server-draft
server-draft:
	OPTIONS="--buildDrafts" docker-compose up server 
