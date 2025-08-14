# Kubernetes 실습

> Play with Kubernetes 온라인 환경 상 실습
> 

## 사전 수행 사항 :  최초 경고 문구의 1,2번 명령어 복사 후 실행

## 실습 1 : Pod 실습

### Pod 생성

---

```
kubectl run nginxserver --image=nginx:latest --port 80
```

### 현재 Pod 목록 확인

---

```
kubectl get pods
---
# 더 자세한 pods 정보 확인
kubectl describe pods
```

- READEY : Pod 총 개수 중 준비된 것
- STATUS
    - `Pending` : 생성중

## 실습 2 : Deployment

### Deployment 생성

```
kubectl create deployment {deployment name} --image=nginx --replicas={number of replica}
```

### Deployment 목록 확인

```
kubectl get deployments.apps
```

- Ready : 관리 중인 pod 수 중 준비된 pod의 개수
- Deployment를 생성 수 pod 목록을 확인해보면 관리하겠다고 정의한 레플리카의 수 만큼 pod의 수가 늘어나있다


### ReplicaSet 수정

```
kubectl edit deployments.apps {deployment name}
```

- 위의 실행 결과에서 내용을 수정하고 저장하면 해당 설정대로 변경된다