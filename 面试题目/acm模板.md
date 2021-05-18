## ACM模板代码片段

#### 1.声明头文件

```C++
#include <iostream> 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <vector>
#include <list>
#define rep(i,m,n) for(int i=m;i<=n;i++)
#define rsp(it,s) for(set<int>::iterator it=s.begin();it!=s.end();it++)
const int inf_int = 2e9;
const long long inf_ll = 2e18;
#define inf_add 0x3f3f3f3f
#define MOD 1000000007
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define pi acos(-1.0)
#define pii pair<int,int>
#define Lson L, mid, rt<<1
#define Rson mid+1, R, rt<<1|1
const int maxn=5e2+10;
using namespace std;
typedef  vector<int> vi;
typedef  long long ll;
typedef  unsigned long long  ull; 
inline int read(){int ra,fh;char rx;rx=getchar(),ra=0,fh=1;
while((rx<'0'||rx>'9')&&rx!='-')rx=getchar();if(rx=='-')
fh=-1,rx=getchar();while(rx>='0'&&rx<='9')ra*=10,ra+=rx-48,
rx=getchar();return ra*fh;}
//#pragma comment(linker, "/STACK:102400000,102400000")
ll gcd(ll p,ll q){return q==0?p:gcd(q,p%q);}
ll qpow(ll p,ll q){ll f=1;while(q){if(q&1)f=f*p;p=p*p;q>>=1;}return f;}
int dir[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
const int N = 1e7+5;

int main()
{
    return 0;   
} 
```

#### 2.搜索

###### 2.1BFS

```C++
bool check(int xx,int yy)
{
    //条件 
    if(xx>=n||xx<0||yy>=m||yy<0||map[xx][yy]||map[xx/2][yy/2]||vis[xx][yy])
        return true;
    return false;
}


void bfs()
{
    queue <NODE> que;
    NODE cur,next;
    cur.x = sx;
    cur.y = sy;
    cur.step = 0;

    vis[cur.x][cur.y] = 1;
    que.push(cur);
    while(!que.empty())
    {
        cur = que.front();
        que.pop();
        if(cur.x==ex&&cur.y == ey)
        {
            ct++;
            return ;
        }
        //分别朝不同方向走 
        for(int i=0;i<8;i++)
        {
            next = cur;
            next.x += dir[i][0];
            next.y += dir[i][1];
            if(check(next.x,next.y))
                continue;
            //步数加一 
            next.step +=1;

            vis[next.x][next.y] = 1;//标记已经走过 
            que.push(next);
        }
    }
    //ans = -1;
}
```

#### 3.图论

###### 3,1LCA最近公共节点

```C++
#include<bits/stdc++.h>
using namespace std;

const int maxn = 40000 + 10;
const int inf = 0x3f3f3f3f;
const double eps = 1e-8;
const double pi = acos(-1.0);
const double ee = exp(1.0);

int head[maxn];
int edgeNum;
struct Edge
{
    int fr, to, next;
    int val;
} e[maxn << 1];

void initEdge()
{
    memset(head, -1, sizeof(head));
    edgeNum = 0;
}

void addEdge(int fr, int to, int val)
{
    e[edgeNum].fr = fr;
    e[edgeNum].to = to;
    e[edgeNum].val = val;
    e[edgeNum].next = head[fr];
    head[fr] = edgeNum++;

    e[edgeNum].fr = to;
    e[edgeNum].to = fr;
    e[edgeNum].val = val;
    e[edgeNum].next = head[to];
    head[to] = edgeNum++;
}

bool vis[maxn];
int dis[maxn];          //根节点到当前点的距离
int ver[maxn << 1];     //dfs遍历时节点的编号
int dep[maxn << 1];     //dfs遍历时节点的深度
int R[maxn];            //dfs遍历时第一次出现当前节点时的遍历序号
int tot;                //下标计数器

void dfs(int u, int d)
{
    vis[u] = true;
    ver[++tot] = u;
    R[u] = tot;
    dep[tot] = d;

    for (int i = head[u]; i != -1; i = e[i].next)
    {
        if (!vis[e[i].to])
        {
            int v = e[i].to;
            int val = e[i].val;
            dis[v] = dis[u] + val;
            dfs(v, d + 1);
            ver[++tot] = u;
            dep[tot] = d;
        }
    }
}


int minDepVerIndex[maxn << 1][20];
void queryInit(int n)
{
    ////////////////////////////
    for (int i = 1; i <= n; i++)
    {
        minDepVerIndex[i][0] = i;
    }
    ////////////////////////////

    for (int j = 1; (1 << j) <= n; j++)
    {
        for (int i = 1; i + (1 << j) - 1 <= n; i++)
        {
            int p = (1 << (j - 1));
            int u = minDepVerIndex[i][j - 1];
            int v = minDepVerIndex[i + p][j - 1];
            minDepVerIndex[i][j] = dep[u] < dep[v] ? u : v;
        }
    }
}

int queryMin(int l, int r)
{
    int k = log2((double)(r - l + 1));
    int u = minDepVerIndex[l][k];
    int v = minDepVerIndex[r - (1 << k) + 1][k];
    return dep[u] < dep[v] ? u : v;
}


//先求出两个点的lca，然后他们之间的最短距离就是一个点走到他们的lca，然后再走向另一个点。
//对应的计算方法就是根节点到lca点的disLca，然后根节点到u点的disU，到v点的disV，
//他们呢间的距离就是disU + disV - 2 * disLCA。


int lca(int u, int v)
{
    int l = R[u];
    int r = R[v];
    if (l > r)
        swap(l, r);
    int index = queryMin(l, r);
    return ver[index];
}

int main()
{
//    freopen("data.txt", "r", stdin);

    int n, q;
    scanf("%d%d", &n, &q);
    initEdge();
    for (int i = 1; i < n; i++)
    {
        int fr, to, val;
        scanf("%d%d%d", &fr, &to, &val);
        addEdge(fr, to, val);
    }

    memset(vis, false, sizeof(vis));
    tot = 0;
    dis[1] = 0;
    dfs(1, 1);

    queryInit((n << 1) - 1);
    while (q--)
    {
        int u, v;
        scanf("%d%d", &u, &v);
        int rt = lca(u, v);
        printf("%d\n", dis[u] + dis[v] - 2 * dis[rt]);
    }

    return 0;
}
```

###### 3.2最短路

```C++
/*  
单源最短路径，Dijkstra算法，邻接矩阵形式，复杂度为O(n^2)  * 
求出源beg到所有点的最短路径，传入图的顶点数，和邻接矩阵cost[][]  *
返回各点的最短路径lowcost[], 路径pre[].pre[i]记录beg到i路径上的父结点，pre[beg]=-1  * 可更改路径权类型，但是权值必须为非负  *  
*/ 
const ll MAXN=100005; 
const ll INF=0x3f3f3f3f;//防止后面溢出，这个不能太大 
bool vis[MAXN]; 
int pre[MAXN]; 
void Dijkstra(ll cost[][MAXN],ll lowcost[],ll n,ll beg) 
{  
    for(int i=0;i<n;i++)  
    {   
        lowcost[i]=INF;
        vis[i]=false;
        pre[i]=-1;  
    }
    lowcost[beg]=0;  
    for(int j=0;j<n;j++)  
    {   
        int k=-1;   
        int Min=INF;   
        for(int i=0;i<n;i++)    
            if(!vis[i]&&lowcost[i]<Min)    
            {     
                Min=lowcost[i];     
                k=i;    
            }   
        if(k==-1) break;   
        vis[k]=true;   
        for(int i=0;i<n;i++)    
            if(!vis[i]&&lowcost[k]+cost[k][i]<lowcost[i])    
            {     
                lowcost[i]=lowcost[k]+cost[k][i]; 
                pre[i]=k;    
            }  
    } 
}
```

###### 3.3优先队列优化的Dijkstra O(E*log(E))

```C++
int m,n;//n is the node   m is the edge
const int MAXN=1e6+5;;
const int MAXM =1e6+5;
struct node{
    int x,d;
    node(){}
    node(int a,int b){x=a;d=b;}
    bool operator < (const node & a) const
    {
        if(d==a.d) return x<a.x;
        else return d > a.d;
    }
};


class Dijkstra_queue{
public:
    void init(){
        for(int i=0;i<=n;i++)
            eg[i].clear();
        for(int i=0;i<=n;i++)
            dist[i]=INF;
    }
    void Run(int s)
    {
        dist[s]=0;
        //用优先队列优化
        priority_queue<node> q;
        q.push(node(s,dist[s]));
        while(!q.empty())
        {
            node x=q.top();q.pop();
            for(int i=0;i<eg[x.x].size();i++)
            {
                node y=eg[x.x][i];
                if(dist[y.x]>x.d+y.d)
                {
                    dist[y.x]=x.d+y.d;
                    q.push(node(y.x,dist[y.x]));
                }
            }
        }
    }
    void addEdge(int u,int v,int w)
    {
        eg[u].push_back(node(v,w));
    }
public:
    int dist[MAXN];
private:
    vector<node> eg[MAXN];//如果MAXN非常大，就把其放到类的外面
};
```

###### 3.4最小生成树

```C++
const int MAXN=110010;//最大点数
const int MAXM=1001000;//最大边 注意范围
int F[MAXN];//并查集使用

struct Edge
{
    int u,v;
    double w;
}edge[MAXM];//储存边的信息，包括起点/终点/权值

int tol=0;//边数，加边前赋值为0



void addedge(int u,int v,double w)
{
    edge[tol].u=u;
    edge[tol].v=v;
    edge[tol++].w=w;
}
void init()
{
    tol=0;
}

bool cmp(Edge a,Edge b)//排序函数，边按照权值从小到大排序
{
    return a.w<b.w;
}

int Find(int x)
{
    if(F[x]==-1)
        return x;
    else
        return F[x]=Find(F[x]);
}

double Kruskal(int n)//传入点数，返回最小生成树的权值，如果不连通返回-1
{
    memset(F,-1,sizeof(F));
    sort(edge,edge+tol,cmp);
    int cnt=0;//计算加入的边数
    double ans=0;
    for(int i=0;i<tol;i++)
    {
        int    u=edge[i].u;
        int    v=edge[i].v;
        double w=edge[i].w;
        int t1=Find(u);
        int t2=Find(v);
        if(t1!=t2)
        {

            ans+=w;
            F[t1]=t2;
            cnt++;
        }
        if(cnt==n-1)
            break;
    }
    if(cnt<n-1)
        return -1;//不连通
    else
        return ans;
}
```

###### 3.5拓扑排序

> 对一个DAG进行拓扑排序有两种方法，广度优先搜索和深度优先搜索。 
>
> 这里介绍广度优先搜索，进行拓扑排序时，每次可以拿出的顶点一定是入度为0的点，即没有被指向的点，因为这样的点表示的事件没有依赖，在一个入度为0的点表示的事件执行完之后，它所指向的顶点所依赖的点就少了一个，所以我们可以先将所有入度为0的点加入一个队列中，然后依次将它们所指向的点的入度减1，再将入度变为0的点也依次加入队列中，这样最后就可以得到一个拓扑有序的序列。

```C++
const int MAXN = 510 ;
const int MAXM = 10000 ;


struct Edge
{
    int from, to,next;
};
Edge edge[MAXM];
int head[MAXN],edgenum;

void init()
{
    edgenum = 0;
    memset(head, -1, sizeof(head));
}

void addEdge(int u, int v)
{
    Edge E1 = {u, v, head[u]};
    edge[edgenum] = E1;
    head[u] = edgenum++;
}

int degree[MAXN];//保存入度

int main()
{
//    freopen("data.txt","r",stdin);
//    ios_base::sync_with_stdio(false);
    int n,m;

    while(cin >> n>>m)
    {
        init();
        memset(degree,0,sizeof(degree));
        //input
        for(int i=0;i<m;i++)
        {
            int u,v;
            cin >> u>>v;
            addEdge(u, v);
            degree[v]++;
        }

        priority_queue<int,vector<int>,greater<int> >q;
        for(int i=1;i<=n;i++)
            if(degree[i]==0)
            q.push(i);

        bool first=1;
        while(!q.empty())
        {
            int cur=q.top();
            q.pop();
            if(first)
            {
                cout<<cur;
                first=0;
            }
            else
                cout<<" "<<cur;
            for(int i=head[cur];i!=-1;i=edge[i].next)
            {
                int now = edge[i].to;
                degree[now]--;//相连的点的入度减1
                if(degree[now]==0)//如果入度为0，加入队列
                    q.push(now);
            }

        }
        cout<<endl;
    }
    return 0;
}

```

###### 3.6无向图强连通分量SCC

```C++
const int maxn = 10005;
int n,m;
vector<int> G[maxn];
int pre[maxn];
int lowlink[maxn];
int sccno[maxn];//记录i所在的scc编号
int dfs_clock,scc_cnt;//scc_cnt  强联通分量个数,++scc_cnt 
stack<int> S;

void dfs(int u)
{
    pre[u] = lowlink[u] = ++dfs_clock;
    S.push(u);
    for(int i = 0 ; i < G[u].size(); i++ )
    {
        int v = G[u][i];
        if(!pre[v])
        {
            dfs(v);
            lowlink[u] = min(lowlink[u],lowlink[v]);
        }
        else if(!sccno[v])
        {
            lowlink[u] = min( lowlink[u],pre[v] );
        }
    }

    if(lowlink[u] == pre[u] )
    {
        scc_cnt++;
        for(;;)
        {
            int x = S.top();
            S.pop();
            sccno[x] = scc_cnt;
            if(x==u) break;
        }
    }
}

void find_scc(int n)
{
    dfs_clock = scc_cnt = 0;
    memset(sccno,0,sizeof(sccno));
    memset(pre,0,sizeof(pre));
    for(int i = 0;i < n ; i++)
    {
        if(!pre[i])
            dfs(i);
    }
}





int ans[maxn];
int main()
{
//    freopen("data.txt","r",stdin);
    ios_base::sync_with_stdio(false);
    cin >> n>>m;
    int u,v;
    while(m--)
    {
        cin >> u>>v;
        u--,v--;
        G[u].push_back(v);
    }
    find_scc(n);
    for(int i=0;i<n;i++)
    {
        ans[sccno[i]] ++;
    }
    int re = 0;
    for(int i=0;i<n;i++)
    {
        if(ans[i]>1) re++;
    }
    cout<<re<<endl;
    return 0;
}

```

#### 4.网络流

###### 4.1最大流

```C++
#include <cstdio>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <sstream>
#include <queue>
#include <typeinfo>
#include <fstream>
#include <map>
#include <stack>
typedef long long ll;
using namespace std;
//freopen("D.in","r",stdin);
//freopen("D.out","w",stdout);
#define sspeed ios_base::sync_with_stdio(0);cin.tie(0)
#define maxn 5000
#define mod 10007
#define eps 1e-9
int Num;
const int inf=0x3f3f3f3f;
inline ll read()
{
    ll x=0,f=1;char ch=getchar();
    while(ch<'0'||ch>'9'){if(ch=='-')f=-1;ch=getchar();}
    while(ch>='0'&&ch<='9'){x=x*10+ch-'0';ch=getchar();}
    return x*f;
}
//**************************************************************************************
namespace NetFlow
{
    const int MAXN=100000,MAXM=100000,inf=1e9;
    struct Edge
    {
        int v,c,f,nx;
        Edge() {}
        Edge(int v,int c,int f,int nx):v(v),c(c),f(f),nx(nx) {}
    } E[MAXM];
    int G[MAXN],cur[MAXN],pre[MAXN],dis[MAXN],gap[MAXN],N,sz;
    void init(int _n) //初始化 
    {
        N=_n,sz=0; memset(G,-1,sizeof(G[0])*N);
    }
    void link(int u,int v,int c)//连接两个点 
    {
        E[sz]=Edge(v,c,0,G[u]); G[u]=sz++;
        E[sz]=Edge(u,0,0,G[v]); G[v]=sz++;
    }
    int ISAP(int S,int T)
    {//S -> T
        int maxflow=0,aug=inf,flag=false,u,v;
        for (int i=0;i<N;++i)cur[i]=G[i],gap[i]=dis[i]=0;
        for (gap[S]=N,u=pre[S]=S;dis[S]<N;flag=false)
        {
            for (int &it=cur[u];~it;it=E[it].nx)
            {
                if (E[it].c>E[it].f&&dis[u]==dis[v=E[it].v]+1)
                {
                    if (aug>E[it].c-E[it].f) aug=E[it].c-E[it].f;
                    pre[v]=u,u=v; flag=true;
                    if (u==T)
                    {
                        for (maxflow+=aug;u!=S;)
                        {
                            E[cur[u=pre[u]]].f+=aug;
                            E[cur[u]^1].f-=aug;
                        }
                        aug=inf;
                    }
                    break;
                }
            }
            if (flag) continue;
            int mx=N;
            for (int it=G[u];~it;it=E[it].nx)
            {
                if (E[it].c>E[it].f&&dis[E[it].v]<mx)
                {
                    mx=dis[E[it].v]; cur[u]=it;
                }
            }
            if ((--gap[dis[u]])==0) break;
            ++gap[dis[u]=mx+1]; u=pre[u];
        }
        return maxflow;
    }
    bool bfs(int S,int T)
    {
        static int Q[MAXN]; memset(dis,-1,sizeof(dis[0])*N);
        dis[S]=0; Q[0]=S;
        for (int h=0,t=1,u,v,it;h<t;++h)
        {
            for (u=Q[h],it=G[u];~it;it=E[it].nx)
            {
                if (dis[v=E[it].v]==-1&&E[it].c>E[it].f)
                {
                    dis[v]=dis[u]+1; Q[t++]=v;
                }
            }
        }
        return dis[T]!=-1;
    }
    int dfs(int u,int T,int low)
    {
        if (u==T) return low;
        int ret=0,tmp,v;
        for (int &it=cur[u];~it&&ret<low;it=E[it].nx)
        {
            if (dis[v=E[it].v]==dis[u]+1&&E[it].c>E[it].f)
            {
                if (tmp=dfs(v,T,min(low-ret,E[it].c-E[it].f)))
                {
                    ret+=tmp; E[it].f+=tmp; E[it^1].f-=tmp;
                }
            }
        }
        if (!ret) dis[u]=-1; return ret;
    }
    int dinic(int S,int T)
    {
        int maxflow=0,tmp;
        while (bfs(S,T))
        {
            memcpy(cur,G,sizeof(G[0])*N);
            while (tmp=dfs(S,T,inf)) maxflow+=tmp;
        }
        return maxflow;
    }
}
using namespace NetFlow;
//map<pair<int,int> ,int> H;
//int tot=1;
/*int get_id(int x,int y)
{
    pair<int,int> A;
    A.first = x;
    A.second = y;
    if(H[A]!=0)
        return H[A];
    else H[A]=tot++;
    return H[A];
}*/

int beg ,end;//起点和终点 
int main()
{
    init(10000);
    int n=read(),f=read(),d=read();
    //定义起点终点 
    beg = 0;
    end = 2*n+d+f+1;
    //加边 
    for(int i=1;i<=f;i++)
        link(beg,i,1);//连接起点和food 
    for(int i=1;i<=d;i++)
        link(i+2*n+f,end,1);//连接终点和drink  
    for(int i=1;i<=n;i++)
        link(f+i,f+i+n,1);//连接两个牛 
    for(int i=1;i<=n;i++)
    {
        int ff=read(),dd=read();
        for(int j=1;j<=ff;j++)
        {
            int x=read();
            link(x,f+i,1);//连接牛和food 
        }

        for(int j=1;j<=dd;j++)
        {
            int x=read();
            link(f+i+n,x+f+2*n,1);//连接牛和drink 
        }
    }
    //得到结果 
    printf("%d\n",dinic(beg,end));//输出结果 
}
```

###### 4.2最大流sap算法

```C++
#include<algorithm>
#include<cstdio>
#include<iostream>
#include<cstring>
#include<cstdlib>
#define N 100020
#define ll long long

using namespace std;

const int MAXN = 100010;//点数的最大值
const int MAXM = 400010;//边数的最大值
const int INF = 0x3f3f3f3f;

struct Edge {
    int to,next,cap,flow;
} edge[MAXM]; //注意是MAXM
int tol;
int head[MAXN];
int gap[MAXN],dep[MAXN],cur[MAXN];
int n,m;

void init() {
    tol = 0;
    memset(head,-1,sizeof(head));
}

void addedge(int u,int v,int w,int rw = 0) {//加边,如果双向边，则4个参数，否则3个参数 
    edge[tol].to = v;
    edge[tol].cap = w;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    head[u] = tol++;
    edge[tol].to = u;
    edge[tol].cap = rw; 
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    head[v] = tol++;
}

int Q[MAXN];

void BFS(int start,int end) {
    memset(dep,-1,sizeof(dep));
    memset(gap,0,sizeof(gap));
    gap[0] = 1;
    int front = 0, rear = 0;
    dep[end] = 0;
    Q[rear++] = end;
    while(front != rear) {
        int u = Q[front++];
        for(int i = head[u]; i != -1; i = edge[i].next) {
            int v = edge[i].to;
            if(dep[v] != -1)continue;
            Q[rear++] = v;
            dep[v] = dep[u] + 1;
            gap[dep[v]]++;
        }
    }
}
int S[MAXN];

int sap(int start,int end,int n) {
    BFS(start,end);
    memcpy(cur,head,sizeof(head));
    int top = 0;
    int u = start;
    int ans = 0;
    while(dep[start] < n) {
        if(u == end) {
            int Min = INF;
            int inser;
            for(int i = 0; i < top; i++)
                if(Min > edge[S[i]].cap - edge[S[i]].flow) {
                    Min = edge[S[i]].cap - edge[S[i]].flow;
                    inser = i;
                }
            for(int i = 0; i < top; i++) {
                edge[S[i]].flow += Min;
                edge[S[i]^1].flow -= Min;
            }
            ans += Min;
            top = inser;
            u = edge[S[top]^1].to;
            continue;
        }
        bool flag = false;
        int v;
        for(int i = cur[u]; i != -1; i = edge[i].next) {
            v = edge[i].to;
            if(edge[i].cap - edge[i].flow && dep[v]+1 == dep[u]) {
                flag = true;
                cur[u] = i;
                break;
            }
        }
        if(flag) {
            S[top++] = cur[u];
            u = v;
            continue;
        }
        int Min = N;
        for(int i = head[u]; i != -1; i = edge[i].next)
            if(edge[i].cap - edge[i].flow && dep[edge[i].to] < Min) {
                Min = dep[edge[i].to];
                cur[u] = i;
            }
        gap[dep[u]]--;
        if(!gap[dep[u]])return ans;
        dep[u] = Min + 1;
        gap[dep[u]]++;
        if(u != start)u = edge[S[--top]^1].to;
    }
    return ans;
}



int main()  
{
    int t;
    scanf("%d",&t);
    while(t--)
    {
        scanf("%d %d",&n,&m);
        int beg ;//超级起点  
        int end ;//超级汇点     
        int maxx,minn;
        maxx = -INF;
        minn =  INF;
        int l1,l2,c;
        int x,y;
        for(int i=1;i<=n;i++)  
        {  
            scanf("%d %d",&x,&y);
            if(x>maxx)
            {
                maxx = x;
                 end= i;
            }
            if(x < minn)
            {
                minn = x;
                beg = i;
            }   
        }  
        init();  
        for(int i=1;i<=m;i++)  
        {  
            scanf("%d%d%d",&l1,&l2,&c);
            addedge(l1,l2,c);
            addedge(l2,l1,c);
        }  
        //得到结果   
        printf("%d\n",sap(beg,end,n));  
    }  

}  
```

###### 4.3最小费用最大流

```C++
#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<queue>
#include<cstdio>
#include<cstring>
#include<cmath>
#include<map>
#include<stack>
#include<set>
#include<iomanip>
//#define mem(dp,a) memset(dp,a,sizeof(dp))
//#define fo(i,n) for(int i=0;i<(n);i++)
//#define INF 0x3f3f3f3f
#define fread() freopen("data.txt","r",stdin)
#define fwrite() freopen("out.out","w",stdout)
using namespace std;
typedef  long long ll;



//最小费用最大流,求最大费用只需要取相反数,结果取相反数即可。
//点的总数为 N,点的编号 0~N-1

const int MAXN = 505;
const int MAXM = 100005;//要比题目给的大
const int INF = 0x3f3f3f3f;
struct Edge
{
    int to, next, cap, flow, cost;
    int x, y;
} edge[MAXM],HH[MAXN],MM[MAXN];
int head[MAXN],tol;
int pre[MAXN],dis[MAXN];
bool vis[MAXN];
int N, M;

void init(int n)
{
    N = n;
    tol = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v, int cap, int cost)//左端点，右端点，容量，花费
{
    edge[tol]. to = v;
    edge[tol]. cap = cap;
    edge[tol]. cost = cost;
    edge[tol]. flow = 0;
    edge[tol]. next = head[u];
    head[u] = tol++;
    edge[tol]. to = u;
    edge[tol]. cap = 0;
    edge[tol]. cost = -cost;
    edge[tol]. flow = 0;
    edge[tol]. next = head[v];
    head[v] = tol++;
}
bool spfa(int s, int t)
{
    queue<int>q;
    for(int i = 0; i < N; i++)
    {
        dis[i] = INF;
        vis[i] = false;
        pre[i] = -1;
    }
    dis[s] = 0;
    vis[s] = true;
    q.push(s);
    while(!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = false;
        for(int i = head[u]; i != -1; i = edge[i]. next)
        {
            int v = edge[i]. to;
            if(edge[i]. cap > edge[i]. flow &&
                    dis[v] > dis[u] + edge[i]. cost )
            {
                dis[v] = dis[u] + edge[i]. cost;
                pre[v] = i;
                if(!vis[v])
                {
                    vis[v] = true;
                    q.push(v);
                }
            }
        }
    }
    if(pre[t] == -1) return false;
    else return true;
}
/*
    * 直接调用获取最小费用和最大流
    * 输入: start-源点，end-汇点（编号从0开始）
    * 返回值: pair<int,int> 第一个是最小费用，第二个是最大流
*/
pair<int, int> minCostMaxflow(int s, int t)
{
    int flow = 0;
    int cost = 0;
    while(spfa(s,t))
    {
        int Min = INF;
        for(int i = pre[t]; i != -1; i = pre[edge[i^1]. to])
        {
            if(Min > edge[i]. cap - edge[i]. flow)
                Min = edge[i]. cap - edge[i]. flow;
        }
//        int percost=0;
        for(int i = pre[t]; i != -1; i = pre[edge[i^1]. to])
        {
            edge[i]. flow += Min;
            edge[i^1]. flow -= Min;
            cost += edge[i]. cost * Min;
//            percost+=edge[i].cost;
        }

//        if(percost>0){
//            return make_pair(cost, flow);
//        }

        cost+=percost*Min;
        flow += Min;

    }
    return make_pair(cost, flow);
}





int a,b,c,d;
int u,v,k;
int m,n;
int main()
{
//    ios_base::sync_with_stdio(false);
//    fread();
    while(~scanf("%d %d",&n,&m))
    {
        init(n+2);
        for(int i=1;i<=n;i++)
        {
            scanf("%d %d %d %d",&a,&b,&c,&d);
            addedge(0,i,b,a);
            addedge(i,n+1,d,-c);
        }

        for(int i=0;i<m;i++)
        {
            scanf("%d %d %d",&u,&v,&k);
            addedge(u,v,INF,k);
            addedge(v,u,INF,k);
        }

        printf("%d\n",-minCostMaxflow(0,n+1).first);
    }
    return 0;
}
```

###### 4.4ZKW最小费用最大流

```C++
struct Edge
{
    int to, next, cap, flow, cost;
    Edge(int _to = 0, int _next = 0, int _cap = 0, int _flow = 0, int _cost = 0) :
            to(_to), next(_next), cap(_cap), flow(_flow), cost(_cost) {}
}edge[MAXM];

struct MinCostMaxFlow
{

    int INFF = 1e9;
    int head[MAXM], tot;//
    int cur[MAXM];//
    int dis[MAXM];
    bool vis[MAXM];
    stringstream s11;
    string rree;
    string tt_;
    vi road_t;
    int ss, tt, N;//源点、汇点和点的总个数（编号是0~N-1）,不需要额外赋值，调用会直接赋值
    int min_cost, max_flow;
    stack<int> S,SR;
    int resnum = 0;
    int ff;
    int flow_min  =  INFF;



    void init()
    {
        tot = 0;
        memset(head, -1, sizeof(head));
    }
    void addedge(int u, int v, int cap, int cost)
    {
        edge[tot] = Edge(v, head[u], cap, 0, cost);
        head[u] = tot++;
        edge[tot] = Edge(u, head[v], 0, 0, -cost);
        head[v] = tot++;
    }
    int aug(int u, int flow)
    {
        if (u == tt) return flow;
        vis[u] = true;
        for (int i = cur[u];i != -1;i = edge[i].next)
        {
            int v = edge[i].to;
            if (edge[i].cap > edge[i].flow && !vis[v] && dis[u] == dis[v] + edge[i].cost)
            {
                int tmp = aug(v, min(flow, edge[i].cap - edge[i].flow));
                edge[i].flow += tmp;
                edge[i ^ 1].flow -= tmp;
                cur[u] = i;
                if (tmp)return tmp;
            }
        }
        return 0;
    }
    bool modify_label()
    {
        int d = INF;
        for (int u = 0;u < N;u++)
            if (vis[u])
                for (int i = head[u];i != -1;i = edge[i].next)
                {
                    int v = edge[i].to;
                    if (edge[i].cap>edge[i].flow && !vis[v])
                        d = min(d, dis[v] + edge[i].cost - dis[u]);
                }
        if (d == INF)return false;
        for (int i = 0;i < N;i++)
            if (vis[i])
            {
                vis[i] = false;
                dis[i] += d;

            }
        return true;
    }
    /*
    * 直接调用获取最小费用和最大流
    * 输入: start-源点，end-汇点，n-点的总个数（编号从0开始）
    * 返回值: pair<int,int> 第一个是最小费用，第二个是最大流
    */
    pair<int, int> mincostmaxflow(int start, int end, int n)
    {
        ss = start, tt = end, N = n;
        min_cost = max_flow = 0;
        for (int i = 0;i < n;i++)dis[i] = 0;
        while (1)
        {
            for (int i = 0;i < n;i++)cur[i] = head[i];
            while (1)
            {
                for (int i = 0;i < n;i++) vis[i] = false;
                int tmp = aug(ss, INF);
                if (tmp == 0)break;
                max_flow += tmp;
                min_cost += tmp*dis[ss];
            }
            if (!modify_label())break;
        }
        return make_pair(min_cost, max_flow);
    }

    void print_one_road(int x,int minn)
    {
        if(ff) return ;
        if(x==tt)
        {
            flow_min = minn;
            SR = S;
            ff = 1;
            return ;
        }
        for (int i = head[x];i != -1;i = edge[i].next)
        {
            if(i%2)
                continue;
            int v = edge[i].to;

            if (edge[i].flow > 0)
            {
                S.push(i);
                print_one_road(v,min(minn,edge[i].flow));
                S.pop();
                break;
            }
        }
        return ;
    }
    void print_road()
    {
        int tedge;
        rree.clear();
        flow_min=INFF;
        ff = 0;

        print_one_road(ss,INFF);

        while(flow_min!=INFF)
        {
            road_t.clear();
            while(!SR.empty())
            {
                tedge = SR.top();
                edge[tedge].flow-=flow_min;
                road_t.push_back(edge[tedge].to-1);
                SR.pop();
            }

            for(int i=road_t.size()-1;i>=1;i--)
            {
                s11.clear();
                s11<< road_t[i];
                s11>>tt_;
                rree += tt_;
                rree+=" ";
            }
            s11.clear();
            s11<< edge_map[road_t[1]]-1;
            s11>>tt_;
            rree += tt_;
            rree+=" ";

            s11.clear();
            s11<< flow_min;
            s11>>tt_;
            rree += tt_;
            rree+="\n";

            while(!S.empty())
                S.pop();

            resnum++;
            ff = 0;
            flow_min=INFF;
            road_t.clear();
            print_one_road(ss,INFF);

        }
    }
    int get_resnum()
    {
        return resnum;
    }

    string get_rree()
    {
        return rree;
    }

}solve;

```

###### 4.5很快dinic最大流

```C++
#include<bits/stdc++.h>
#define MAXN 410
#define MAXM 9000+10
#define INF 0x3f3f3f3f
using namespace std;

struct Edge
{
    int from, to, cap, flow, next;
};
Edge edge[MAXM];
int head[MAXN], cur[MAXN], edgenum;
int dist[MAXN];
bool vis[MAXN];
int N, M,ss,tt;
void init()
{
    edgenum = 0;
    memset(head, -1, sizeof(head));
}
void addEdge(int u, int v, int w)
{
    Edge E1 = {u, v, w, 0, head[u]};
    edge[edgenum] = E1;
    head[u] = edgenum++;
    Edge E2 = {v, u, 0, 0, head[v]};
    edge[edgenum] = E2;
    head[v] = edgenum++;
}

bool BFS(int s, int t)
{
    queue<int> Q;
    memset(dist, -1, sizeof(dist));
    memset(vis, false, sizeof(vis));
    dist[s] = 0;
    vis[s] = true;
    Q.push(s);
    while(!Q.empty())
    {
        int u = Q.front();
        Q.pop();
        for(int i = head[u]; i != -1; i = edge[i].next)
        {
            Edge E = edge[i];
            if(!vis[E.to] && E.cap > E.flow)
            {
                dist[E.to] = dist[u] + 1;
                if(E.to == t) return true;
                vis[E.to] = true;
                Q.push(E.to);
            }
        }
    }
    return false;
}
int DFS(int x, int a, int t)
{
    if(x == t || a == 0) return a;
    int flow = 0, f;
    for(int &i = cur[x]; i != -1; i = edge[i].next)
    {
        Edge &E = edge[i];
        if(dist[E.to] == dist[x] + 1 && (f = DFS(E.to, min(a, E.cap - E.flow), t)) > 0)
        {
            edge[i].flow += f;
            edge[i^1].flow -= f;
            flow += f;
            a -= f;
            if(a == 0) break;
        }
    }
    return flow;
}
int Maxflow(int s, int t)
{
    int flow = 0;
    while(BFS(s, t))
    {
        memcpy(cur, head, sizeof(head));
        flow += DFS(s, INF, t);
    }
    return flow;
}

void solve()
{
//    Maxflow(ss, tt);
//
//    for(int i = 0; i < edgenum; i+=2)
//    {
//        Edge E = edge[i];
//        if(E.cap == E.flow)
//        {
//            edge[i].cap = 1;
//            edge[i].flow = 0;
//        }
//        else
//        {
//            edge[i].cap = INF;
//            edge[i].flow = 0;
//        }
//        edge[i^1].cap = edge[i^1].flow = 0;
//    }
    printf("%d\n",  Maxflow(ss, tt)%1001);
}
int T;
int main()
{
//    freopen("data.txt","r",stdin);
    scanf("%d", &T);
    while(T--)
    {
        scanf("%d%d", &N, &M);
        scanf("%d%d",&ss,&tt);
        init();
        int u,v,w;
        while(M--)
        {
            scanf("%d%d%d", &u,&v,&w);
            addEdge(u, v, w*1001+1);
        }
        solve();
    }
    return 0;
}
```

###### 4.6KM

```C++
//KM算法求二分图的最佳完美匹配
struct KM {
    int slack[N],res[N];
    int l[N],r[N],lx[N],rx[N],g[N][N];

    void clear(int n) {
        for(int i=1;i<=n;i++) {
            res[i]=0;
            for(int j=1;j<=n;j++) g[i][j]=-1;
        }
    }
    bool find(int x,int n) {
        lx[x]=1;
        for(int i=1;i<=n;i++)
            if(!rx[i]&&g[x][i]!=-1) {
                int tmp=g[x][i]-l[x]-r[i];
                if(!tmp) {
                    rx[i]=1;
                    if(!res[i]||find(res[i],n)) {
                        res[i]=x;
                        return 1;
                    }
                } else
                    slack[i]=min(slack[i],tmp);
            }
        return 0;
    }
    int solve(int n) {
        if(!n) return 0;
        for(int i=1;i<=n;i++) r[i]=0;
        for(int i=1;i<=n;i++) {
            l[i]=INF;
            for(int j=1;j<=n;j++) if(g[i][j]!=-1)
                l[i]=min(l[i],g[i][j]);
        }
        for(int i=1;i<=n;i++) {
            for(int j=1;j<=n;j++) slack[j]=INF;
            for(;;) {
                for(int j=1;j<=n;j++) lx[j]=rx[j]=0;
                if(find(i,n)) break;
                int mini=INF;
                for(int i=1;i<=n;i++) if(!rx[i])
                    mini=min(mini,slack[i]);
                for(int i=1;i<=n;i++) {
                    if(lx[i]) l[i]+=mini;
                    if(rx[i]) r[i]-=mini;
                    else slack[i]-=mini;
                }
            }
        }
        int ans=0;
        for(int i=1;i<=n;i++)
            ans+=l[i]+r[i];
        return ans;
    }
} km;
```

#### 5.数据结构

###### 5.1线段树

```C++
const int INFINITE = INT_MAX;
const int MAXNUM = 1000;
struct SegTreeNode
{
    int val;
    int addMark;//延迟标记
}segTree[MAXNUM];//定义线段树

/*
功能：构建线段树
root：当前线段树的根节点下标
arr: 用来构造线段树的数组
istart：数组的起始位置
iend：数组的结束位置
*/
void build(int root, int arr[], int istart, int iend)
{
    segTree[root].addMark = 0;//----设置标延迟记域
    if(istart == iend)//叶子节点
        segTree[root].val = arr[istart];
    else
    {
        int mid = (istart + iend) / 2;
        build(root*2+1, arr, istart, mid);//递归构造左子树
        build(root*2+2, arr, mid+1, iend);//递归构造右子树
        //根据左右子树根节点的值，更新当前根节点的值
        segTree[root].val = min(segTree[root*2+1].val, segTree[root*2+2].val);
    }
}

/*
功能：当前节点的标志域向孩子节点传递
root: 当前线段树的根节点下标
*/
void pushDown(int root)
{
    if(segTree[root].addMark != 0)
    {
        //设置左右孩子节点的标志域，因为孩子节点可能被多次延迟标记又没有向下传递
        //所以是 “+=”
        segTree[root*2+1].addMark += segTree[root].addMark;
        segTree[root*2+2].addMark += segTree[root].addMark;
        //根据标志域设置孩子节点的值。因为我们是求区间最小值，因此当区间内每个元
        //素加上一个值时，区间的最小值也加上这个值
        segTree[root*2+1].val += segTree[root].addMark;
        segTree[root*2+2].val += segTree[root].addMark;
        //传递后，当前节点标记域清空
        segTree[root].addMark = 0;
    }
}

/*
功能：线段树的区间查询
root：当前线段树的根节点下标
[nstart, nend]: 当前节点所表示的区间
[qstart, qend]: 此次查询的区间
*/
int query(int root, int nstart, int nend, int qstart, int qend)
{
    //查询区间和当前节点区间没有交集
    if(qstart > nend || qend < nstart)
        return INFINITE;
    //当前节点区间包含在查询区间内
    if(qstart <= nstart && qend >= nend)
        return segTree[root].val;
    //分别从左右子树查询，返回两者查询结果的较小值
    pushDown(root); //----延迟标志域向下传递
    int mid = (nstart + nend) / 2;
    return min(query(root*2+1, nstart, mid, qstart, qend),
               query(root*2+2, mid + 1, nend, qstart, qend));

}

/*
功能：更新线段树中某个区间内叶子节点的值
root：当前线段树的根节点下标
[nstart, nend]: 当前节点所表示的区间
[ustart, uend]: 待更新的区间
addVal: 更新的值（原来的值加上addVal）
*/
void update(int root, int nstart, int nend, int ustart, int uend, int addVal)
{
    //更新区间和当前节点区间没有交集
    if(ustart > nend || uend < nstart)
        return ;
    //当前节点区间包含在更新区间内
    if(ustart <= nstart && uend >= nend)
    {
        segTree[root].addMark += addVal;
        segTree[root].val += addVal;
        return ;
    }
    pushDown(root); //延迟标记向下传递
    //更新左右孩子节点
    int mid = (nstart + nend) / 2;
    update(root*2+1, nstart, mid, ustart, uend, addVal);
    update(root*2+2, mid+1, nend, ustart, uend, addVal);
    //根据左右子树的值回溯更新当前节点的值
    segTree[root].val = min(segTree[root*2+1].val, segTree[root*2+2].val);
}
```

```C++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int INF = 1e9;


inline int read(){int ra,fh;char rx;rx=getchar(),ra=0,fh=1;
while((rx<'0'||rx>'9')&&rx!='-')rx=getchar();if(rx=='-')
fh=-1,rx=getchar();while(rx>='0'&&rx<='9')ra*=10,ra+=rx-48,
rx=getchar();return ra*fh;}

const int N = 1000005;

struct {
    int l,r,val,tag;
}t[3000005];

void pushdown(int k)
{
    //向下传递tag
    if(t[k].l==t[k].r) return;

    int tag = t[k].tag;
    t[k].tag=0;

    if(tag)
    {
        t[k<<1].tag =t[k<<1].tag+tag;
        t[k<<1|1].tag=t[k<<1|1].tag+tag;
        t[k<<1].val=t[k<<1].val+tag;
        t[k<<1|1].val=t[k<<1|1].val+tag;
    }
}

void build(int k,int l,int r)
{
    t[k].l = l;
    t[k].r = r;
    if(l==r) return ;
    int mid = (l+r)>>1;
    build(k<<1,l,mid);
    build(k<<1|1,mid+1,r);
}

void update(int k,int x,int y,int val)
{
    pushdown(k);

    int l = t[k].l;
    int r = t[k].r;

    //更新最大值
    if(l==x&&r==y)
    {
        t[k].tag ++;
        t[k].val ++;
        return ;
    }

    //分段更新
    int mid = (l+r)>>1;
    if(y<=mid)
        update(k<<1,x,y,val);
    else if(x>mid)
        update(k<<1|1,x,y,val);
    else
    {
        update(k<<1,x,mid,val);
        update(k<<1|1,mid+1,y,val);
    }

    t[k].val=max(t[k<<1].val,t[k<<1|1].val);
}


int query(int k,int x)
{
    pushdown(k);
    int l = t[k].l;
    int r = t[k].r;
    if(l==r)
    {
        return t[k].val;
    }
    int mid = (l+r)>>1;
    if(x<=mid)
    {
        return query(k<<1,x);
    }
    else
    {
        return query(k<<1|1,x);
    }
}

int n;
int l,r;
int main()
{
//    freopen("data.txt","r",stdin);
    ios_base::sync_with_stdio(false);
    cin >> n;
    build(1,1,1000000);
    for(int i=0;i<n;i++)
    {
        cin >> l>>r;
        update(1,l,r,1);
    }
    pushdown(1);
    cout << t[1].val<<endl;
    return 0;
}

```

###### 5.2带权并查集

```C++
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+7;
long long fa[maxn],a[maxn],b[maxn],ans[maxn],vis[maxn],sum[maxn],n;
int fi(int x){return fa[x]==x?x:fa[x]=fi(fa[x]);}
void uni(int x,int y)
{
    x=fi(x),y=fi(y);
    fa[x]=y;sum[y]+=sum[x];
}
int main()
{
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&a[i]);
        sum[i]=a[i];
        fa[i]=i;
    }
    for(int i=1;i<=n;i++)scanf("%d",&b[i]);
    long long tmp = 0;
    for(int i=n;i>1;i--)
    {
        vis[b[i]]=1;
        if(vis[b[i]-1])uni(b[i]-1,b[i]);
        if(vis[b[i]+1])uni(b[i]+1,b[i]);
        tmp=max(sum[fi(b[i])],tmp);
        ans[i-1]=tmp;
    }
    for(int i=1;i<=n;i++)
        cout<<ans[i]<<endl;
}
```

###### 5.3优先队列

```C++
 priority_queue<int, vector<int>, greater<int> > qi2;//从小到大的优先级队列，可将greater改为less，即为从大到小

```

###### 5.40-1字典树

```C++
#include<bits/stdc++.h>  
typedef long long ll;  
using namespace std;  
inline int read()  
{  
    int x=0,f=1;char ch=getchar();  
    while(ch<'0'||ch>'9'){if(ch=='-')f=-1;ch=getchar();}  
    while(ch>='0'&&ch<='9'){x=x*10+ch-'0';ch=getchar();}  
    return x*f;  
}  

int bin[35];  
int n,ans,cnt;  
int val[100005],last[100005];  
struct edge{  
    int to,next,v;  
}e[200005];  


//01字典树  
struct trie{  
    int cnt;  
    int ch[3000005][2];//zis = 30 * MAXN  
    void insert(int x)  
    {  
        int now=0;  
        for(int i=30;i>=0;i--)  
        {  
            int t=x&bin[i];t>>=i;  
            if(!ch[now][t])  ch[now][t]=++cnt;  
            now=ch[now][t];  
        }  
    }  

    void query(int x)  
    {  
        int tmp=0,now=0;//当前为x 寻找另一个y 使得x^y最大  
        for(int i=30;i>=0;i--)  
        {  
            int t=x&bin[i];t>>=i;  
            if(ch[now][t^1])now=ch[now][t^1],tmp+=bin[i];  
            else now=ch[now][t];  
        }  
        ans=max(tmp,ans);  
    }  
}trie;  

void insert(int u,int v,int w)  
{  
    e[++cnt].to=v;e[cnt].next=last[u];last[u]=cnt;e[cnt].v=w;  
    e[++cnt].to=u;e[cnt].next=last[v];last[v]=cnt;e[cnt].v=w;  
}  

void dfs(int x,int fa)  
{  
    for(int i=last[x];i;i=e[i].next)  
        if(e[i].to!=fa)  
        {  
            val[e[i].to]=val[x]^e[i].v;  
            dfs(e[i].to,x);  
        }  
}  

int main()  
{  
    bin[0]=1;for(int i=1;i<=30;i++)bin[i]=bin[i-1]<<1;  

    n=read();  
    for(int i=1;i<n;i++)  
    {  
        int u=read(),v=read(),w=read();  
        insert(u,v,w);  
    }  
    dfs(1,0);  

    for(int i=1;i<=n;i++)  
        trie.insert(val[i]);  
    for(int i=1;i<=n;i++)  
        trie.query(val[i]);  
    printf("%d",ans);  
    return 0;  
}  
```

#### 6.数论

###### 6.1大数模板

```C++
#define MAX_L 205 //最大长度，可以修改
class bign
{
public:
    int len, s[MAX_L];//数的长度，记录数组
	//构造函数
    bign();
    bign(const char*);
    bign(int);
    bool sign;//符号 1正数 0负数
    string toStr() const;//转化为字符串，主要是便于输出
    friend istream& operator>>(istream &,bign &);//重载输入流
    friend ostream& operator<<(ostream &,bign &);//重载输出流
	//重载复制
    bign operator=(const char*);
    bign operator=(int);
    bign operator=(const string);
	//重载各种比较
    bool operator>(const bign &) const;
    bool operator>=(const bign &) const;
    bool operator<(const bign &) const;
    bool operator<=(const bign &) const;
    bool operator==(const bign &) const;
    bool operator!=(const bign &) const;
	//重载四则运算
    bign operator+(const bign &) const;
    bign operator++();
    bign operator++(int);
    bign operator+=(const bign&);
    bign operator-(const bign &) const;
    bign operator--();
    bign operator--(int);
    bign operator-=(const bign&);
    bign operator*(const bign &)const;
    bign operator*(const int num)const;
    bign operator*=(const bign&);
    bign operator/(const bign&)const;
    bign operator/=(const bign&);
	//四则运算的衍生运算
    bign operator%(const bign&)const;//取模（余数）
    bign factorial()const;//阶乘
    bign Sqrt()const;//整数开根（向下取整）
    bign pow(const bign&)const;//次方
	//一些乱乱的函数
    void clean();
    ~bign();
};
#define max(a,b) a>b ? a : b
#define min(a,b) a<b ? a : b

bign::bign()
{
    memset(s, 0, sizeof(s));
    len = 1;
    sign = 1;
}

bign::bign(const char *num)
{
    *this = num;
}

bign::bign(int num)
{
    *this = num;
}

string bign::toStr() const
{
    string res;
    res = "";
    for (int i = 0; i < len; i++)
        res = (char)(s[i] + '0') + res;
    if (res == "")
        res = "0";
    if (!sign&&res != "0")
        res = "-" + res;
    return res;
}

istream &operator>>(istream &in, bign &num)
{
    string str;
    in>>str;
    num=str;
    return in;
}

ostream &operator<<(ostream &out, bign &num)
{
    out<<num.toStr();
    return out;
}

bign bign::operator=(const char *num)
{

    memset(s, 0, sizeof(s));
    char a[MAX_L] = "";
    if (num[0] != '-')
        strcpy(a, num);
    else
        for (int i = 1; i < strlen(num); i++)
            a[i - 1] = num[i];
    sign = !(num[0] == '-');
    len = strlen(a);
    for (int i = 0; i < strlen(a); i++)
        s[i] = a[len - i - 1] - 48;
    return *this;
}

bign bign::operator=(int num)
{
    char temp[MAX_L];
    sprintf(temp, "%d", num);
    *this = temp;
    return *this;
}

bign bign::operator=(const string num)
{
    const char *tmp;
    tmp = num.c_str();
    *this = tmp;
    return *this;
}

bool bign::operator<(const bign &num) const
{
    if (sign^num.sign)
        return num.sign;
    if (len != num.len)
        return len < num.len;
    for (int i = len - 1; i >= 0; i--)
        if (s[i] != num.s[i])
            return sign ? (s[i] < num.s[i]) : (!(s[i] < num.s[i]));
    return !sign;
}

bool bign::operator>(const bign&num)const
{
    return num < *this;
}

bool bign::operator<=(const bign&num)const
{
    return !(*this>num);
}

bool bign::operator>=(const bign&num)const
{
    return !(*this<num);
}

bool bign::operator!=(const bign&num)const
{
    return *this > num || *this < num;
}

bool bign::operator==(const bign&num)const
{
    return !(num != *this);
}

bign bign::operator+(const bign &num) const
{
    if (sign^num.sign)
    {
        bign tmp = sign ? num : *this;
        tmp.sign = 1;
        return sign ? *this - tmp : num - tmp;
    }
    bign result;
    result.len = 0;
    int temp = 0;
    for (int i = 0; temp || i < (max(len, num.len)); i++)
    {
        int t = s[i] + num.s[i] + temp;
        result.s[result.len++] = t % 10;
        temp = t / 10;
    }
    result.sign = sign;
    return result;
}

bign bign::operator++()
{
    *this = *this + 1;
    return *this;
}

bign bign::operator++(int)
{
    bign old = *this;
    ++(*this);
    return old;
}

bign bign::operator+=(const bign &num)
{
    *this = *this + num;
    return *this;
}

bign bign::operator-(const bign &num) const
{
    bign b=num,a=*this;
    if (!num.sign && !sign)
    {
        b.sign=1;
        a.sign=1;
        return b-a;
    }
    if (!b.sign)
    {
        b.sign=1;
        return a+b;
    }
    if (!a.sign)
    {
        a.sign=1;
        b=bign(0)-(a+b);
        return b;
    }
    if (a<b)
    {
        bign c=(b-a);
        c.sign=false;
        return c;
    }
    bign result;
    result.len = 0;
    for (int i = 0, g = 0; i < a.len; i++)
    {
        int x = a.s[i] - g;
        if (i < b.len) x -= b.s[i];
        if (x >= 0) g = 0;
        else
        {
            g = 1;
            x += 10;
        }
        result.s[result.len++] = x;
    }
    result.clean();
    return result;
}

bign bign::operator * (const bign &num)const
{
    bign result;
    result.len = len + num.len;

    for (int i = 0; i < len; i++)
        for (int j = 0; j < num.len; j++)
            result.s[i + j] += s[i] * num.s[j];

    for (int i = 0; i < result.len; i++)
    {
        result.s[i + 1] += result.s[i] / 10;
        result.s[i] %= 10;
    }
    result.clean();
    result.sign = !(sign^num.sign);
    return result;
}

bign bign::operator*(const int num)const
{
    bign x = num;
    bign z = *this;
    return x*z;
}
bign bign::operator*=(const bign&num)
{
    *this = *this * num;
    return *this;
}

bign bign::operator /(const bign&num)const
{
    bign ans;
    ans.len = len - num.len + 1;
    if (ans.len < 0)
    {
        ans.len = 1;
        return ans;
    }

    bign divisor = *this, divid = num;
    divisor.sign = divid.sign = 1;
    int k = ans.len - 1;
    int j = len - 1;
    while (k >= 0)
    {
        while (divisor.s[j] == 0) j--;
        if (k > j) k = j;
        char z[MAX_L];
        memset(z, 0, sizeof(z));
        for (int i = j; i >= k; i--)
            z[j - i] = divisor.s[i] + '0';
        bign dividend = z;
        if (dividend < divid) { k--; continue; }
        int key = 0;
        while (divid*key <= dividend) key++;
        key--;
        ans.s[k] = key;
        bign temp = divid*key;
        for (int i = 0; i < k; i++)
            temp = temp * 10;
        divisor = divisor - temp;
        k--;
    }
    ans.clean();
    ans.sign = !(sign^num.sign);
    return ans;
}

bign bign::operator/=(const bign&num)
{
    *this = *this / num;
    return *this;
}

bign bign::operator%(const bign& num)const
{
    bign a = *this, b = num;
    a.sign = b.sign = 1;
    bign result, temp = a / b*b;
    result = a - temp;
    result.sign = sign;
    return result;
}

bign bign::pow(const bign& num)const
{
    bign result = 1;
    for (bign i = 0; i < num; i++)
        result = result*(*this);
    return result;
}

bign bign::factorial()const
{
    bign result = 1;
    for (bign i = 1; i <= *this; i++)
        result *= i;
    return result;
}

void bign::clean()
{
    if (len == 0) len++;
    while (len > 1 && s[len - 1] == '\0')
        len--;
}

bign bign::Sqrt()const
{
    if(*this<0)return -1;
    if(*this<=1)return *this;
    bign l=0,r=*this,mid;
    while(r-l>1)
    {
        mid=(l+r)/2;
        if(mid*mid>*this)
            r=mid;
        else
            l=mid;
    }
    return l;
}

bign::~bign()
{
}
```

###### 6.2素数筛

```C++
int isprime[1000005];
int prime[1000005];
int sum[1000005];
int cnt  = 0;
void initprime()
{
    for(int i=2;i<N;i++)
    {
        isprime[i] = true;
    }
    for(int i=2;i<N;i++)
    {
        if(isprime[i])
        {
            prime[++cnt]=i;
            for(int j = i<< 1;j<N;j+=i)
            {
                isprime[j] = false;
                sum[j] += i;//在筛素数时就找出其和 
            }
        }   
    } 
}
```

###### 6.3欧拉函数O(sqrt(n))

```C++
//计算欧拉函数O(sqrt(n))
int Phi(int x)
{
    int i,re=x;
    for(i=2;i*i<=x;i++)
        if(x%i==0)
        {
            re/=i;re*=i-1;
            while(x%i==0)
                x/=i;
        }
    if(x^1) re/=x,re*=x-1;
    return re;
}
```

###### 6.4矩阵快速幂+快速斐波那契

```C++
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace std;

const int MOD = 10000;

struct matrix {     //矩阵 
    long long m[2][2];
}ans;

matrix base = {1, 1, 1, 0}; 

matrix multi(matrix a, matrix b) {  //矩阵相乘，返回一个矩阵 
    matrix tmp;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            tmp.m[i][j] = 0;
            for(int k = 0;  k < 2; k++)
                tmp.m[i][j] = (tmp.m[i][j] + a.m[i][k] * b.m[k][j]) % MOD;
        }
    }
    return tmp;
}

int matrix_pow(matrix a, int n) {   //矩阵快速幂，矩阵a的n次幂 
    ans.m[0][0] = ans.m[1][1] = 1;  //初始化为单位矩阵 
    ans.m[0][1] = ans.m[1][0] = 0;
    while(n) {
        if(n & 1) ans = multi(ans, a);
        a = multi(a, a);
        n >>= 1;
    }
    return ans.m[0][1];
}

int main() {
    int n;
    while(scanf("%d", &n), n != -1) {
        printf("%d\n", matrix_pow(base, n));
    }
    return 0;
} 
```

###### 6.5矩阵快速幂

```C++
#include<stdio.h>
#include<iostream>
#include<cmath>
#include<stdlib.h>
#include<string>
#include<cstring>

#define MOD 1000000007
typedef long long ll;
using namespace std;
const int N = 4;


//f[n] = f[n-1] + 5*f[n-2] + f[n-3] + f[n-4];

ll n ;

struct Matrix{
    ll mat[N][N];
    Matrix operator*(const Matrix& m)const{
        Matrix tmp;
        for(int i = 0 ; i < N ; i++){
            for(int j = 0 ; j < N ; j++){
                tmp.mat[i][j] = 0;
                for(int k = 0 ; k < N ; k++){
                    tmp.mat[i][j] += mat[i][k]*m.mat[k][j]%MOD;
                    tmp.mat[i][j] %= MOD;
                }
            }
        }
        return tmp;
    }
};

ll Pow(Matrix &m){
    if(n==1)
    {
        return 1;
    }

    if(n ==2)
    {
        return 5;
    }
    if(n == 3 )
    {
        return 11;
    }
    if(n == 4)
    {
        return 36;
    }
    n -= 4;
    Matrix ans;
    memset(ans.mat , 0 , sizeof(ans.mat));
    for(int i = 0 ; i < N ; i++)
        ans.mat[i][i] = 1;
    while(n){
        if(n&1)
            ans = ans*m;
        n >>= 1;
        m = m*m;
    }
    //初始值
    ll sum = 0;
    sum += ans.mat[0][0]*36%MOD;
    sum += ans.mat[0][1]*11%MOD;
    sum += ans.mat[0][2]*5%MOD;
    sum += ans.mat[0][3]*1%MOD;
    return sum%MOD;
}

int main(){
    Matrix m;
    while(scanf("%lld" , &n ) != EOF){
        memset(m.mat , 0 , sizeof(m.mat));
        m.mat[0][0] = 1;
        m.mat[0][1] = 5;
        m.mat[0][2] = 1;
        m.mat[0][3] = -1;
        m.mat[1][0] = m.mat[2][1] = m.mat[3][2] = 1;
        printf("%lld\n" , (Pow(m)+MOD)%MOD );
    }
    return 0;
}
```

###### 6.6矩阵模板

```C++
const int  N  = 50;
struct Matrix {
    int siz;
    int a[N][N];
    Matrix(int sizt) {
        siz = sizt;
        memset(a, 0, sizeof(a));
    }

    Matrix operator * (const Matrix y) {
        Matrix ans(siz);
        for(int i = 0; i < siz; i++)
            for(int j = 0; j < siz; j++)
                for(int k = 0; k < siz; k++)
                    ans.a[i][j] += a[i][k]*y.a[k][j];

        for(int i = 0; i < siz; i++)
            for(int j = 0; j < siz; j++)
                ans.a[i][j] %= MOD;
        return ans;
    }


    Matrix operator + (const Matrix y) {
        Matrix ans(siz);
        for(int i = 0; i < siz; i++)
            for(int j = 0; j < siz; j++)
                    ans.a[i][j] = a[i][j]+y.a[i][j];

        for(int i = 0; i < siz; i++)
            for(int j = 0; j < siz; j++)
                ans.a[i][j] %= MOD;

        return ans;
    }

    void operator = (const Matrix b) {
        siz = b.siz;
        for(int i = 0; i < siz; i++)
            for(int j = 0; j < siz; j++)
                a[i][j] = b.a[i][j];
    }

    void print()
    {
        for(int i=0;i<siz;i++)
        {
            for(int j=0;j<siz;j++)
            {
                cout<<a[i][j]<<" ";
            }
            cout<<endl;
        }
    }
};

Matrix matpow(Matrix b, int k){
    if(k == 1){
        return b;
    }

    Matrix res = Matrix(b.siz);
    for(int i=0;i<res.siz;i++){
        res.a[i][i] = 1;
    }

    if(k==0)  return res;
    while(k){
        if(k&1) res = res*b;
        b = b*b;
        k >>= 1;
    }
    return res;
}
```

###### 6.7斯特林数&&母函数

```C++
//ll fac[N]={1};
//ll stir1[N][N];
//ll stir2[N][N];
//ll mu[2][N];
//int f=0;

//ll C[N][N];
//void init(){


//    //卡特兰数
//    f[1]=1;
//    for(int i=2;i<=n;i++)
//        f[i] = f[i-1]*(4*i-2) / (i+1);


    //母函数，用于求拆分数
//    for(int i=0;i<N;i++){
//        mu[0][i]=1;
//        mu[1][i]=0;
//    }
//    for(int i=2;i<N;i++){
//        for(int j=0;j<N;j++)
//            for(int k=0;k+j<N;k+=i)
//                mu[!f][k+j]+=mu[f][j];
//        for(int k=0;k<N;k++)
//            mu[f][k]=0;
//        f=!f;
//    }

    //!n
//    for(int i=1;i<N;i++)
//        fac[i]= ( (fac[i-1]%MOD) * (i%MOD) )%MOD;
    //斯特林数1
//    for(int i=1;i<N;i++){
//        stir1[i][0]=0;
//        stir1[i][i]=1;
//       for(int j=1;j<i;j++)
//           stir1[i][j]= (  stir1[i-1][j-1] + ( ((i-1) % MOD) * (stir1[i-1][j] % MOD) )%MOD  )%MOD;
//    }


    //斯特林数2
//    for(int i=1;i<N;i++){
//        stir2[i][0]=0;
//        stir2[i][i]=1;
//       for(int j=1;j<i;j++)
//           stir2[i][j]= (  stir2[i-1][j-1] + ( ((j) % MOD) * (stir2[i-1][j] % MOD) )%MOD  )%MOD;
//    }

    //组合数
//    C[1][0] = C[1][1] = 1;
//    for (int i = 2; i < N; i++){
//        C[i][0] = 1;
//        for (int j = 1; j < N; j++)
//            C[i][j] = (C[i - 1][j] + C[i - 1][j - 1])%MOD;
//    }

//}
```

###### 6.8质数分解

```C++
ll ktl[N];

int n,p,not_prime[N],prime[N],tot,low[N],s[N];

void init(int n)
{
    for (int i=2;i<=n;i++)
    {
        if (!not_prime[i])
        {
            prime[++tot]=i;low[i]=i;
        }
        for (int j=1;j<=tot&&prime[j]*i<=n;j++)
        {
            not_prime[prime[j]*i]=1;
            low[i*prime[j]]=prime[j];
            if (i%prime[j]==0) break;
        }
    }
}

void solve(int x,int y)//质数分解
{
    while (x>1)
    {
        s[low[x]]+=y;
        x/=low[x];
    }
}

int main() {
     ios_base::sync_with_stdio(false);
//    freopen("data.txt","r",stdin);

    while(cin >> n >> p)
    {
        init(n*2);
        for (int i=1;i<=n;i++)     solve(i,-1);
        for (int i=n+2;i<=n*2;i++) solve(i,1); //模拟 c(2n,n)

        ll ans=1;

        for (int i=2;i<=n*2;i++)
            for (int j=1;j<=s[i];j++) ans=(ll)ans*i%p;

        cout<<ans<<endl;
    }
    return 0;
}
```

###### 6.9康托展开

```C++
int  fac[] = {1,1,2,6,24,120,720,5040,40320}; //i的阶乘为fac[i]  
// 康托展开-> 表示数字a是 a的全排列中从小到大排，排第几  
// n表示1~n个数  a数组表示数字。  
int kangtuo(int n,char a[])  
{  
    int i,j,t,sum;  
    sum=0;  
    for( i=0; i<n ;++i)  
    {  
        t=0;  
        for(j=i+1;j<n;++j)  
            if( a[i]>a[j] )  
                ++t;  
        sum+=t*fac[n-i-1];  
    }  
    return sum+1;  
}  

//康托展开
LL Work(char str[])
{
    int len = strlen(str);
    LL ans = 0;
    for(int i=0; i<len; i++)
    {
        int tmp = 0;
        for(int j=i+1; j<len; j++)
            if(str[j] < str[i]) tmp++;
        ans += tmp * f[len-i-1];  //f[]为阶乘
    }
    return ans;  //返回该字符串是全排列中第几大，从1开始
}

```

###### 6.10康托逆展开

```C++
int  fac[] = {1,1,2,6,24,120,720,5040,40320};
//康托展开的逆运算,{1...n}的全排列，中的第k个数为s[]
void reverse_kangtuo(int n,int k,char s[])
{
    int i, j, t, vst[8]={0};
    --k;
    for (i=0; i<n; i++)
    {
        t = k/fac[n-i-1];
        for (j=1; j<=n; j++)
            if (!vst[j])
            {
                if (t == 0) break;
                --t;
            }
        s[i] = '0'+j;
        vst[j] = 1;
        k %= fac[n-i-1];
    }
}



//康托展开逆运算
void Work(LL n,LL m)
{
    n--;
    vector<int> v;
    vector<int> a;
    for(int i=1;i<=m;i++)
        v.push_back(i);
    for(int i=m;i>=1;i--)
    {
        LL r = n % f[i-1];
        LL t = n / f[i-1];
        n = r;
        sort(v.begin(),v.end());
        a.push_back(v[t]);
        v.erase(v.begin()+t);
    }
    vector<int>::iterator it;
    for(it = a.begin();it != a.end();it++)
        cout<<*it;
    cout<<endl;
}
```

#### 7.字符串

###### 7.1KMP

```C++
const int N = 1000005;
void kmp_pre(char x[],int m,int next[])
{   int i,j;  j=next[0]=-1;  i=0;
    while(i<m)
    {
        while(-1!=j && x[i]!=x[j])
        j=next[j];   next[++i]=++j;
    }
}



int KMP_Count(char x[],int m,char y[],int n)
{
    int next1[N*2];
    memset(next1,0,sizeof(next1));
    int i,j;
    int ans=0;
    kmp_pre(x,m,next1);
    i=j=0;
    while(i<n)
    {
        while(-1!=j && y[i]!=x[j])  j=next1[j];
        i++;j++;
        if(j>=m)
        {    ans++;    j=next1[j];   }
    }
    return ans;
}
```

###### 7.2后缀数组

```C++
/*
    Problem: JZOJ1598(询问一个字符串中有多少至少出现两次的子串)
    Content: SA's Code and Explanation
    Author : YxuanwKeith
*/

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int MAXN = 100005;

char ch[MAXN], All[MAXN];
int SA[MAXN], rank[MAXN], Height[MAXN], tax[MAXN], tp[MAXN], a[MAXN], n, m; 
char str[MAXN];
//rank[i] 第i个后缀的排名; SA[i] 排名为i的后缀位置; Height[i] 排名为i的后缀与排名为(i-1)的后缀的LCP
//tax[i] 计数排序辅助数组; tp[i] rank的辅助数组(计数排序中的第二关键字),与SA意义一样。
//a为原串
void RSort() {
    //rank第一关键字,tp第二关键字。
    for (int i = 0; i <= m; i ++) tax[i] = 0;
    for (int i = 1; i <= n; i ++) tax[rank[tp[i]]] ++;
    for (int i = 1; i <= m; i ++) tax[i] += tax[i-1];
    for (int i = n; i >= 1; i --) SA[tax[rank[tp[i]]] --] = tp[i]; //确保满足第一关键字的同时，再满足第二关键字的要求
} //计数排序,把新的二元组排序。

int cmp(int *f, int x, int y, int w) { return f[x] == f[y] && f[x + w] == f[y + w]; } 
//通过二元组两个下标的比较，确定两个子串是否相同

void Suffix() {
    //SA
    for (int i = 1; i <= n; i ++) rank[i] = a[i], tp[i] = i;
    m = 127 ,RSort(); //一开始是以单个字符为单位，所以(m = 127)

    for (int w = 1, p = 1, i; p < n; w += w, m = p) { //把子串长度翻倍,更新rank

        //w 当前一个子串的长度; m 当前离散后的排名种类数
        //当前的tp(第二关键字)可直接由上一次的SA的得到
        for (p = 0, i = n - w + 1; i <= n; i ++) tp[++ p] = i; //长度越界,第二关键字为0
        for (i = 1; i <= n; i ++) if (SA[i] > w) tp[++ p] = SA[i] - w;

        //更新SA值,并用tp暂时存下上一轮的rank(用于cmp比较)
        RSort(), swap(rank, tp), rank[SA[1]] = p = 1;

        //用已经完成的SA来更新与它互逆的rank,并离散rank
        for (i = 2; i <= n; i ++) rank[SA[i]] = cmp(tp, SA[i], SA[i - 1], w) ? p : ++ p;
    }
    //离散：把相等的字符串的rank设为相同。
    //LCP
    int j, k = 0;
    for(int i = 1; i <= n; Height[rank[i ++]] = k) 
        for( k = k ? k - 1 : k, j = SA[rank[i] - 1]; a[i + k] == a[j + k]; ++ k);
    //这个知道原理后就比较好理解程序
}

void Init() {
    scanf("%s", str);
    n = strlen(str);
    for (int i = 0; i < n; i ++) a[i + 1] = str[i];
}

int main() {
    Init();
    Suffix();

    int ans = Height[2];
    for (int i = 3; i <= n; i ++) ans += max(Height[i] - Height[i - 1], 0);
    printf("%d\n", ans);    
}
```

```C++
/*
 * SPOJ 694
 * 给定一个字符串，求不相同子串个数。
 * 每个子串一定是某个后缀的前缀，那么原问题等价于求所有后缀之间的不相同子串个数。
 * 总数为n*(n-1)/2,再减掉height[i]的和就是答案
 */

#include <iostream>
#include <string.h>
#include <algorithm>
#include <stdio.h>
using namespace std;
const int MAXN=1010;

/*
*suffix array
*倍增算法  O(n*logn)
*待排序数组长度为n,放在0~n-1中，在最后面补一个0
*build_sa( ,n+1, );//注意是n+1;
*getHeight(,n);
*例如：
*n   = 8;
*num[]   = { 1, 1, 2, 1, 1, 1, 1, 2, $ };注意num最后一位为0，其他大于0
*rank[]  = { 4, 6, 8, 1, 2, 3, 5, 7, 0 };rank[0~n-1]为有效值，rank[n]必定为0无效值
*sa[]    = { 8, 3, 4, 5, 0, 6, 1, 7, 2 };sa[1~n]为有效值，sa[0]必定为n是无效值
*height[]= { 0, 0, 3, 2, 3, 1, 2, 0, 1 };height[2~n]为有效值
*
*/

int sa[MAXN];//SA数组，表示将S的n个后缀从小到大排序后把排好序的
             //的后缀的开头位置顺次放入SA中
int t1[MAXN],t2[MAXN],c[MAXN];//求SA数组需要的中间变量，不需要赋值
int rank[MAXN],height[MAXN];
//待排序的字符串放在s数组中，从s[0]到s[n-1],长度为n,且最大值小于m,
//除s[n-1]外的所有s[i]都大于0，r[n-1]=0
//函数结束以后结果放在sa数组中
void build_sa(int s[],int n,int m)
{
    int i,j,p,*x=t1,*y=t2;
    //第一轮基数排序，如果s的最大值很大，可改为快速排序
    for(i=0;i<m;i++)c[i]=0;
    for(i=0;i<n;i++)c[x[i]=s[i]]++;
    for(i=1;i<m;i++)c[i]+=c[i-1];
    for(i=n-1;i>=0;i--)sa[--c[x[i]]]=i;
    for(j=1;j<=n;j<<=1)
    {
        p=0;
        //直接利用sa数组排序第二关键字
        for(i=n-j;i<n;i++)y[p++]=i;//后面的j个数第二关键字为空的最小
        for(i=0;i<n;i++)if(sa[i]>=j)y[p++]=sa[i]-j;
        //这样数组y保存的就是按照第二关键字排序的结果
        //基数排序第一关键字
        for(i=0;i<m;i++)c[i]=0;
        for(i=0;i<n;i++)c[x[y[i]]]++;
        for(i=1;i<m;i++)c[i]+=c[i-1];
        for(i=n-1;i>=0;i--)sa[--c[x[y[i]]]]=y[i];
        //根据sa和x数组计算新的x数组
        swap(x,y);
        p=1;x[sa[0]]=0;
        for(i=1;i<n;i++)
            x[sa[i]]=y[sa[i-1]]==y[sa[i]] && y[sa[i-1]+j]==y[sa[i]+j]?p-1:p++;
        if(p>=n)break;
        m=p;//下次基数排序的最大值
    }
}
void getHeight(int s[],int n)
{
    int i,j,k=0;
    for(i=0;i<=n;i++)rank[sa[i]]=i;
    for(i=0;i<n;i++)
    {
        if(k)k--;
        j=sa[rank[i]-1];
        while(s[i+k]==s[j+k])k++;
        height[rank[i]]=k;
    }
}

char str[MAXN];
int s[MAXN];

int main()
{
    //freopen("in.txt","r",stdin);
    //freopen("out.txt","w",stdout);
    int T;
    scanf("%d",&T);
    while(T--)
    {
        scanf("%s",str);
        int n=strlen(str);
        for(int i=0;i<=n;i++)s[i]=str[i];
        build_sa(s,n+1,128);
        getHeight(s,n);
        int ans=n*(n+1)/2;
        for(int i=2;i<=n;i++)ans-=height[i];
        printf("%d\n",ans);
    }
    return 0;
}
```

###### 7.3AC自动机

```C++
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <vector>
#include <list>
#define rep(i,m,n) for(int i=m;i<=n;i++)
#define rsp(it,s) for(set<int>::iterator it=s.begin();it!=s.end();it++)
const int inf_int = 2e9;
const long long inf_ll = 2e18;
#define inf_add 0x3f3f3f3f
#define MOD 1000000007
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define pi acos(-1.0)
#define pii pair<int,int>
#define Lson L, mid, rt<<1
#define Rson mid+1, R, rt<<1|1
const int maxn=5e2+10;
using namespace std;
typedef  vector<int> vi;
typedef  long long ll;
typedef  unsigned long long  ull;
inline int read(){int ra,fh;char rx;rx=getchar(),ra=0,fh=1;
    while((rx<'0'||rx>'9')&&rx!='-')rx=getchar();if(rx=='-')
        fh=-1,rx=getchar();while(rx>='0'&&rx<='9')ra*=10,ra+=rx-48,
                                                  rx=getchar();return ra*fh;}
//#pragma comment(linker, "/STACK:102400000,102400000")
ll gcd(ll p,ll q){return q==0?p:gcd(q,p%q);}
ll qpow(ll p,ll q){ll f=1;while(q){if(q&1)f=f*p;p=p*p;q>>=1;}return f;}
int dir[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
const int N = 1e6+5;
const int M = 27;

class Trie
{

public:
    //next数组存储树
    //fail数组存储下一个要匹配的节点号
    //end数组主要是用来标记一个模式串结尾
    int next[N][M],fail[N],end[N];
    int root,L;
    int newnode()
    {
        for(int i = 0;i < M;i++)//每一个节点对应0-128中的任意一个。
            next[L][i] = -1;
        end[L++] = 0;//表示下面没有节点 初始化，如果是记录次数，就赋0 还可以赋任意的数，
        return L-1;
    }
    void init()
    {
        L = 0;
        root = newnode();
    }
    void insert(char s[],int id)
    {
        int len = strlen(s);
        int now = root;
        for(int i = 0;i < len;i++)
        {
            int k =s[i]-'a';
            if(next[now][k] == -1)
                next[now][k] = newnode();
            now=next[now][k];
        }
//        end[now]=id;//记录当前匹配单词的节点
        end[now]++;//也可以用匹配单词结束后来记录次数
    }

    //BFS求fail
    void build()
    {
        queue<int>Q;
        fail[root] = root;
        //初始化root及其子节点
        for(int i = 0;i < M;i++)
            if(next[root][i] == -1)
                next[root][i] = root;
            else
            {
                fail[next[root][i]] = root;
                Q.push(next[root][i]);
            }
        while(!Q.empty())
        {
            int now = Q.front();
            Q.pop();
            //遍历节点
            for(int i = 0;i < M;i++)
                if(next[now][i] == -1)
                    //不需要失配函数，对所有转移一视同仁
                    next[now][i] = next[fail[now]][i];
                else
                {
                    fail[next[now][i]] = next[fail[now]][i];
                    Q.push(next[now][i]);
                }
        }
    }
    void query(char buf[])
    {
        int ans=0;
        int len = strlen(buf);
        int now = root;
        bool flag = false;
        for(int i = 0;i < len;i++)
        {
            int k =buf[i]-'a';
            now = next[now][k];
            int temp = now;
            //其会匹配多个模式串
            while(temp != root)
            {
                ans+=end[temp];
                end[temp] = 0;
                temp = fail[temp];
            }
        }
        printf("%d\n",ans);
    }
};

char buf[1000005];

Trie ac;
int T;

int main()
{
//    freopen("data.txt","r",stdin);
    int n,m;
    scanf("%d",&T);
    for(int k=0;k<T;k++)
    {
        scanf("%d",&n);
        ac.init();
        for(int i = 1;i <= n;i++)
        {
            scanf("%s",buf);
            ac.insert(buf,i);
        }
        ac.build();
        scanf("%s",buf);
        ac.query(buf);
    }
    return 0;
}
```

###### 7.4Manacher算法

```C++
#include<iostream>
#include<cstring>
#include<cstdio>
using namespace std;

const int MAXN =111111;
char Ma[MAXN*2];
int Mp[MAXN*2];
void Manacher(char s[],int len){
    int l=0;
    Ma[l++]='$';
    Ma[l++]='#';
    for(int i=0;i<len;i++){
        Ma[l++]=s[i];
        Ma[l++]='#';
    }
    Ma[l]=0;
    int mx=0;
    int id=0;
    for(int i=0;i<l;i++){
        Mp[i]=mx>i?min(Mp[2*id-1],mx-1):1;
        while(Ma[i+Mp[i]]==Ma[i-Mp[i]])
        Mp[i]++;
        if(i+Mp[i]>mx)
        {
            mx=i+Mp[i];
            id=i;
        }
    }
}
/*
* abaaba
* i:
0 1 2 3 4 5 6 7 8 9 10 11 12 13
* Ma[i]: $ # a # b # a # a $ b # a #
* Mp[i]: 1 1 2 1 4 1 2 7 2 1 4 1 2 1
*/

char s[MAXN];
int main()
{
//    freopen("data.txt","r",stdin);
    while(scanf("%s",s)==1){
        int len=strlen(s);
        Manacher(s,len);
        int ans=0;
        int N=0;
        for(int i=0;i<2*len+2;i++)
        {
            if(Mp[i]-1>ans)
            {
                ans=Mp[i];                //ans即为最长回文子串的长度
                N=i;
            }
        }
        int st=(N-ans+1)/2,end=st+ans-1;
        for(int i=st;i<end;i++)
        {
            printf("%c",s[i]);
        }
        printf("\n");
    }
    return 0;
}

```

#### 8.动态规划

###### 8.1数位DP

```C++
//数字[L,R]中，round number数字的个数。round number即数字转换成二进制后0的个数大于等于1的个数。
//加前导0判断
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
int digit[35],dp[35][35][35][2],pos;
int dfs(int l,int cnt1,int cnt2,bool zero,bool jud){
    if(l==0) return cnt1>=cnt2;
    if(!jud&&dp[l][cnt1][cnt2][zero]!=-1) return dp[l][cnt1][cnt2][zero];
    int nex = jud ? digit[l] : 1;
    int ans = 0;
    for(int i=0;i <= nex;i++){
        ans += dfs( l-1 , zero ? 0 : cnt1+(i==0) ,cnt2+(i==1) , zero&&(i==0) , jud&&i==nex );
    }
    if(!jud)dp[l][cnt1][cnt2][zero]=ans;
    return ans;
}
int f(int num){
    pos = 0;
    while(num){
        digit[++pos]=num%2;
        num>>=1;
    }
    return dfs(pos,0,0,true,true);
}
int main(){
    #ifdef LOCAL
    freopen("in.txt","r",stdin);
    freopen("out.txt","w",stdout);
    #endif
    memset(dp,-1,sizeof(dp));
    int m,n;
    cin>>m>>n;
    cout<<f(n)-f(m-1)<<endl;
    return 0;
}
```

```C++
//[L,R]中，不含4或62的数字个数。
//dp[l][six]：l为数字长度，six为最后一位数字是否为6。
//dfs(int l,bool six,bool jud)，jud判断是否为边界。
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
int digit[10],dp[10][2],vis[10][2];
int dfs(int l,bool six,bool jud){
    if(l==0) return 1;
    if(!jud&&vis[l][six])return dp[l][six];
    int len = jud ? digit[l] : 9;
    int nes = 0;
    for(int i=0;i<=len;i++){
        if((i==4)||(six&&i==2)) continue;
    nes += dfs(l-1 , i==6 , jud&&(i==len));
    }
    if(!jud){
        vis[l][six]=true;
        dp[l][six]=nes;
    }
    return nes;
}
int f(int k){
    memset(dp,0,sizeof(dp));
    memset(vis,0,sizeof(vis));
    int pos = 0;
    while(k){
        digit[++pos]=k%10;
        k=k/10;
    }
    int ans = dfs(pos,false,true);
    return ans;
}
int main(){
    #ifdef LOCAL
    freopen("in.txt","r",stdin);
    freopen("out.txt","w",stdout);
    #endif
int m,n;
    while(scanf("%d%d",&m,&n)&&(m+n)){
        cout<<f(n)-f(m-1)<<endl;
    }
return 0;
}
```

###### 8.2递增子序列LIS

```C++
/最长严格递增
//LIS O(n*log(n));
int getLISLength(int length) {
    vector<int> ivec;
    for(int i = 0; i < length; ++i)
    {
        if (ivec.size() == 0 || ivec.back() < num[i])
            ivec.push_back(num[i]);
        else
        {
            int low = lower_bound(ivec.begin(),ivec.end(),num[i])-ivec.begin();
            ivec[low] = num[i];
        }
    }
    return ivec.size();
}
//最长非递减
//LIS O(n*log(n));
int getLISLength( int length)
{
    vector<ll> ivec;
    ivec.clear();
    for(int i = 0; i < length; ++i)
    {
        if (ivec.size() == 0 || ivec.back() <= num[i])
            ivec.push_back(num[i]);
        else
        {
            int low = upper_bound(ivec.begin(),ivec.end(),num[i])-ivec.begin();
            //找到大于等于num[i]的数
            ivec[low] = num[i];
        }
    }
    return ivec.size();
}
```

```C++
const int maxn =5005;
int a[maxn];//序列 从0-(n-1)
const ll INF = 1e18;
int d[maxn];//以i结尾的最长上升子序列长度
ll g[maxn];//d值为i的最小值
//最长严格递增子序列
//LIS O(n*log(n));
int getLISLength(int lenth)
{
    for(int i=1;i<=n;i++)
    {
        g[i]=INF;
    }
    for(int i=0;i<n;i++)
    {
        int k = lower_bound(g+1,g+n+1,a[i]) - g;
        d[i]=k;
        g[k]=a[i];
    }

    return (lower_bound(g+1,g+n+1,INF) - g)-1;
}

//最长非递减子序列
//LIS O(n*log(n));
int getLISLength(int lenth)
{
    for(int i=1;i<=n;i++)
    {
        g[i]=INF;
    }
    for(int i=0;i<n;i++)
    {
        int k = upper_bound(g+1,g+n+1,a[i]) - g;
        d[i]=k;
        g[k]=a[i];
    }

    return (lower_bound(g+1,g+n+1,INF) - g)-1;
}

int main()
{
//    freopen("data.txt","r",stdin);
    ios_base::sync_with_stdio(false);

    cin >> n;
    for(int i=0;i<n;i++)
    {
        cin >> a[i];
    }
    cout <<getLISLength(n)<<endl;

    return 0;
}
```

#### 9.bitset

```C++
#include<bitset>
biset<32> s(10);  //32位的bitset,赋值为十进制的10
bitset<32> bs("011010101001"); //用字符串初始化
bs[20]=1; //像数值一样对某一位进行操作

s=10; //赋值为十进制的10
s.reset();//清零
s.set(); //全部位置放置为1
s.count(); //统计1的个数

b.flip(); //把b中所有二进制位逐位取反
b.to_ulong();//用b中同样的二进制位返回一个unsigned long值
```

#### 10.二分

###### 10.1lower_bound

```C++
//算法返回一个非递减序列[first, last)中的第一个大于等于值val的位置。
//这个算法中，first是最终要返回的位置
int lower_bound(int *array, int size, int key)
{
    int first = 0, middle;
    int half, len;
    len = size;

    while(len > 0) {
        half = len >> 1;
        middle = first + half;
        if(array[middle] < key) {     
            first = middle + 1;          
            len = len-half-1;       //在右边子序列中查找
        }
        else
            len = half;            //在左边子序列（包含middle）中查找
    }
    return first;
}
```

###### 10.2upper_bound

```C++
// 算法返回一个非递减序列[first, last)中第一个大于val的位置。
int upper_bound(int *array, int size, int key)
{
    int first = 0, len = size-1;
    int half, middle;

    while(len > 0){
        half = len >> 1;
        middle = first + half;
        if(array[middle] > key)     //中位数大于key,在包含last的左半边序列中查找。
            len = half;
        else{
            first = middle + 1;    //中位数小于等于key,在右半边序列中查找。
            len = len - half - 1;
        }
    }
    return first;
}
```

###### 10.3广义二分

```C++
int ans=-1;
int l = 0;
int r = maxx;

while(l<=r)
{
    int mid=(l+r)>>1;
    if(check(mid))
        ans=mid,r=mid-1;
    else
        l=mid+1;
}
cout << ans<<endl;
```

#### 11.几何

###### 11.1计算几何

```C++
/*
计算几何

目录 
㈠ 点的基本运算 
1. 平面上两点之间距离 1 
2. 判断两点是否重合 1 
3. 矢量叉乘 1 
4. 矢量点乘 2 
5. 判断点是否在线段上 2 
6. 求一点饶某点旋转后的坐标 2 
7. 求矢量夹角 2 

㈡ 线段及直线的基本运算 
1. 点与线段的关系 3 
2. 求点到线段所在直线垂线的垂足 4 
3. 点到线段的最近点 4 
4. 点到线段所在直线的距离 4 
5. 点到折线集的最近距离 4 
6. 判断圆是否在多边形内 5 
7. 求矢量夹角余弦 5 
8. 求线段之间的夹角 5 
9. 判断线段是否相交 6 
10.判断线段是否相交但不交在端点处 6 
11.求线段所在直线的方程 6 
12.求直线的斜率 7 
13.求直线的倾斜角 7 
14.求点关于某直线的对称点 7 
15.判断两条直线是否相交及求直线交点 7 
16.判断线段是否相交，如果相交返回交点 7 

㈢ 多边形常用算法模块 
1. 判断多边形是否简单多边形 8 
2. 检查多边形顶点的凸凹性 9 
3. 判断多边形是否凸多边形 9 
4. 求多边形面积 9 
5. 判断多边形顶点的排列方向，方法一 10 
6. 判断多边形顶点的排列方向，方法二 10 
7. 射线法判断点是否在多边形内 10 
8. 判断点是否在凸多边形内 11 
9. 寻找点集的graham算法 12 
10.寻找点集凸包的卷包裹法 13 
11.判断线段是否在多边形内 14 
12.求简单多边形的重心 15 
13.求凸多边形的重心 17 
14.求肯定在给定多边形内的一个点 17 
15.求从多边形外一点出发到该多边形的切线 18 
16.判断多边形的核是否存在 19 

㈣ 圆的基本运算 
.点是否在圆内 20 
.求不共线的三点所确定的圆 21 

㈤ 矩形的基本运算 
1.已知矩形三点坐标，求第4点坐标 22 

㈥ 常用算法的描述 22 

㈦ 补充 
1．两圆关系： 24 
2．判断圆是否在矩形内： 24 
3．点到平面的距离： 25 
4．点是否在直线同侧： 25 
5．镜面反射线： 25 
6．矩形包含： 26 
7．两圆交点： 27 
8．两圆公共面积： 28 
9. 圆和直线关系： 29 
10. 内切圆： 30 
11. 求切点： 31 
12. 线段的左右旋： 31 
13．公式： 32 
*/

/* 需要包含的头文件 */
#include <cmath >

/* 常用的常量定义 */
const double INF = 1E200 const double EP = 1E-10 const int MAXV = 300 const double PI = 3.14159265

/* 基本几何结构 */
struct POINT
{
    double x;
    double y;
    POINT(double a = 0, double b = 0)
    {
        x = a;
        y = b;
    } //constructor
};
struct LINESEG
{
    POINT s;
    POINT e;
    LINESEG(POINT a, POINT b)
    {
        s = a;
        e = b;
    }
    LINESEG() {}
};
struct LINE // 直线的解析方程 a*x+b*y+c=0  为统一表示，约定 a >= 0
{
    double a;
    double b;
    double c;
    LINE(double d1 = 1, double d2 = -1, double d3 = 0)
    {
        a = d1;
        b = d2;
        c = d3;
    }
};

/**********************
 *                    * 
 *   点的基本运算     * 
 *                    * 
 **********************/

double dist(POINT p1, POINT p2) // 返回两点之间欧氏距离
{
    return (sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)));
}
bool equal_point(POINT p1, POINT p2) // 判断两个点是否重合
{
    return ((abs(p1.x - p2.x) < EP) && (abs(p1.y - p2.y) < EP));
}
/****************************************************************************** 
r=multiply(sp,ep,op),得到(sp-op)和(ep-op)的叉积 
r>0：ep在矢量opsp的逆时针方向； 
r=0：opspep三点共线； 
r<0：ep在矢量opsp的顺时针方向 
*******************************************************************************/
double multiply(POINT sp, POINT ep, POINT op)
{
    return ((sp.x - op.x) * (ep.y - op.y) - (ep.x - op.x) * (sp.y - op.y));
}
/* 
r=dotmultiply(p1,p2,op),得到矢量(p1-op)和(p2-op)的点积，如果两个矢量都非零矢量 
r<0：两矢量夹角为钝角；
r=0：两矢量夹角为直角；
r>0：两矢量夹角为锐角 
*******************************************************************************/
double dotmultiply(POINT p1, POINT p2, POINT p0)
{
    return ((p1.x - p0.x) * (p2.x - p0.x) + (p1.y - p0.y) * (p2.y - p0.y));
}
/****************************************************************************** 
判断点p是否在线段l上
条件：(p在线段l所在的直线上) && (点p在以线段l为对角线的矩形内)
*******************************************************************************/
bool online(LINESEG l, POINT p)
{
    return ((multiply(l.e, p, l.s) == 0) && (((p.x - l.s.x) * (p.x - l.e.x) <= 0) && ((p.y - l.s.y) * (p.y - l.e.y) <= 0)));
}
// 返回点p以点o为圆心逆时针旋转alpha(单位：弧度)后所在的位置
POINT rotate(POINT o, double alpha, POINT p)
{
    POINT tp;
    p.x -= o.x;
    p.y -= o.y;
    tp.x = p.x * cos(alpha) - p.y * sin(alpha) + o.x;
    tp.y = p.y * cos(alpha) + p.x * sin(alpha) + o.y;
    return tp;
}
/* 返回顶角在o点，起始边为os，终止边为oe的夹角(单位：弧度) 
 角度小于pi，返回正值 
 角度大于pi，返回负值 
 可以用于求线段之间的夹角 
原理：
 r = dotmultiply(s,e,o) / (dist(o,s)*dist(o,e))
 r'= multiply(s,e,o)

 r >= 1 angle = 0;
 r <= -1 angle = -PI
 -1<r<1 && r'>0 angle = arccos(r)
 -1<r<1 && r'<=0 angle = -arccos(r)
*/
double angle(POINT o, POINT s, POINT e)
{
    double cosfi, fi, norm;
    double dsx = s.x - o.x;
    double dsy = s.y - o.y;
    double dex = e.x - o.x;
    double dey = e.y - o.y;

    cosfi = dsx * dex + dsy * dey;
    norm = (dsx * dsx + dsy * dsy) * (dex * dex + dey * dey);
    cosfi /= sqrt(norm);

    if (cosfi >= 1.0)
        return 0;
    if (cosfi <= -1.0)
        return -3.1415926;

    fi = acos(cosfi);
    if (dsx * dey - dsy * dex > 0)
        return fi; // 说明矢量os 在矢量 oe的顺时针方向
    return -fi;
}
/*****************************\ 
  *                             * 
  *      线段及直线的基本运算   * 
  *                             * 
  \*****************************/

/* 判断点与线段的关系,用途很广泛 
本函数是根据下面的公式写的，P是点C到线段AB所在直线的垂足 

                AC dot AB 
        r =     --------- 
                 ||AB||^2 
             (Cx-Ax)(Bx-Ax) + (Cy-Ay)(By-Ay) 
          = ------------------------------- 
                          L^2 

    r has the following meaning: 

        r=0      P = A 
        r=1      P = B 
        r<0   P is on the backward extension of AB 
  r>1      P is on the forward extension of AB 
        0<r<1  P is interior to AB 
*/
double relation(POINT p, LINESEG l)
{
    LINESEG tl;
    tl.s = l.s;
    tl.e = p;
    return dotmultiply(tl.e, l.e, l.s) / (dist(l.s, l.e) * dist(l.s, l.e));
}
// 求点C到线段AB所在直线的垂足 P
POINT perpendicular(POINT p, LINESEG l)
{
    double r = relation(p, l);
    POINT tp;
    tp.x = l.s.x + r * (l.e.x - l.s.x);
    tp.y = l.s.y + r * (l.e.y - l.s.y);
    return tp;
}
/* 求点p到线段l的最短距离,并返回线段上距该点最近的点np 
注意：np是线段l上到点p最近的点，不一定是垂足 */
double ptolinesegdist(POINT p, LINESEG l, POINT &np)
{
    double r = relation(p, l);
    if (r < 0)
    {
        np = l.s;
        return dist(p, l.s);
    }
    if (r > 1)
    {
        np = l.e;
        return dist(p, l.e);
    }
    np = perpendicular(p, l);
    return dist(p, np);
}
// 求点p到线段l所在直线的距离,请注意本函数与上个函数的区别
double ptoldist(POINT p, LINESEG l)
{
    return abs(multiply(p, l.e, l.s)) / dist(l.s, l.e);
}
/* 计算点到折线集的最近距离,并返回最近点. 
注意：调用的是ptolineseg()函数 */
double ptopointset(int vcount, POINT pointset[], POINT p, POINT &q)
{
    int i;
    double cd = double(INF), td;
    LINESEG l;
    POINT tq, cq;

    for (i = 0; i < vcount - 1; i++)
    {
        l.s = pointset[i];

        l.e = pointset[i + 1];
        td = ptolinesegdist(p, l, tq);
        if (td < cd)
        {
            cd = td;
            cq = tq;
        }
    }
    q = cq;
    return cd;
}
/* 判断圆是否在多边形内.ptolineseg()函数的应用2 */
bool CircleInsidePolygon(int vcount, POINT center, double radius, POINT polygon[])
{
    POINT q;
    double d;
    q.x = 0;
    q.y = 0;
    d = ptopointset(vcount, polygon, center, q);
    if (d < radius || fabs(d - radius) < EP)
        return true;
    else
        return false;
}
/* 返回两个矢量l1和l2的夹角的余弦(-1 --- 1)注意：如果想从余弦求夹角的话，注意反余弦函数的定义域是从 0到pi */
double cosine(LINESEG l1, LINESEG l2)
{
 return (((l1.e.x-l1.s.x)*(l2.e.x-l2.s.x) + 
 (l1.e.y-l1.s.y)*(l2.e.y-l2.s.y))/(dist(l1.e,l1.s)*dist(l2.e,l2.s))) );
}
// 返回线段l1与l2之间的夹角 单位：弧度 范围(-pi，pi)
double lsangle(LINESEG l1, LINESEG l2)
{
    POINT o, s, e;
    o.x = o.y = 0;
    s.x = l1.e.x - l1.s.x;
    s.y = l1.e.y - l1.s.y;
    e.x = l2.e.x - l2.s.x;
    e.y = l2.e.y - l2.s.y;
    return angle(o, s, e);
}
// 如果线段u和v相交(包括相交在端点处)时，返回true
//
//判断P1P2跨立Q1Q2的依据是：( P1 - Q1 ) × ( Q2 - Q1 ) * ( Q2 - Q1 ) × ( P2 - Q1 ) >= 0。
//判断Q1Q2跨立P1P2的依据是：( Q1 - P1 ) × ( P2 - P1 ) * ( P2 - P1 ) × ( Q2 - P1 ) >= 0。
bool intersect(LINESEG u, LINESEG v)
{
    return ((max(u.s.x, u.e.x) >= min(v.s.x, v.e.x)) && //排斥实验
            (max(v.s.x, v.e.x) >= min(u.s.x, u.e.x)) &&
            (max(u.s.y, u.e.y) >= min(v.s.y, v.e.y)) &&
            (max(v.s.y, v.e.y) >= min(u.s.y, u.e.y)) &&
            (multiply(v.s, u.e, u.s) * multiply(u.e, v.e, u.s) >= 0) && //跨立实验
            (multiply(u.s, v.e, v.s) * multiply(v.e, u.e, v.s) >= 0));
}
//  (线段u和v相交)&&(交点不是双方的端点) 时返回true
bool intersect_A(LINESEG u, LINESEG v)
{
    return ((intersect(u, v)) &&
            (!online(u, v.s)) &&
            (!online(u, v.e)) &&
            (!online(v, u.e)) &&
            (!online(v, u.s)));
}
// 线段v所在直线与线段u相交时返回true；方法：判断线段u是否跨立线段v
bool intersect_l(LINESEG u, LINESEG v)
{
    return multiply(u.s, v.e, v.s) * multiply(v.e, u.e, v.s) >= 0;
}
// 根据已知两点坐标，求过这两点的直线解析方程： a*x+b*y+c = 0  (a >= 0)
LINE makeline(POINT p1, POINT p2)
{
    LINE tl;
    int sign = 1;
    tl.a = p2.y - p1.y;
    if (tl.a < 0)
    {
        sign = -1;
        tl.a = sign * tl.a;
    }
    tl.b = sign * (p1.x - p2.x);
    tl.c = sign * (p1.y * p2.x - p1.x * p2.y);
    return tl;
}
// 根据直线解析方程返回直线的斜率k,水平线返回 0,竖直线返回 1e200
double slope(LINE l)
{
    if (abs(l.a) < 1e-20)
        return 0;
    if (abs(l.b) < 1e-20)
        return INF;
    return -(l.a / l.b);
}
// 返回直线的倾斜角alpha ( 0 - pi)
double alpha(LINE l)
{
    if (abs(l.a) < EP)
        return 0;
    if (abs(l.b) < EP)
        return PI / 2;
    double k = slope(l);
    if (k > 0)
        return atan(k);
    else
        return PI + atan(k);
}
// 求点p关于直线l的对称点
POINT symmetry(LINE l, POINT p)
{
    POINT tp;
    tp.x = ((l.b * l.b - l.a * l.a) * p.x - 2 * l.a * l.b * p.y - 2 * l.a * l.c) / (l.a * l.a + l.b * l.b);
    tp.y = ((l.a * l.a - l.b * l.b) * p.y - 2 * l.a * l.b * p.x - 2 * l.b * l.c) / (l.a * l.a + l.b * l.b);
    return tp;
}
// 如果两条直线 l1(a1*x+b1*y+c1 = 0), l2(a2*x+b2*y+c2 = 0)相交，返回true，且返回交点p
bool lineintersect(LINE l1, LINE l2, POINT &p) // 是 L1，L2
{
    double d = l1.a * l2.b - l2.a * l1.b;
    if (abs(d) < EP) // 不相交
        return false;
    p.x = (l2.c * l1.b - l1.c * l2.b) / d;
    p.y = (l2.a * l1.c - l1.a * l2.c) / d;
    return true;
}
// 如果线段l1和l2相交，返回true且交点由(inter)返回，否则返回false
bool intersection(LINESEG l1, LINESEG l2, POINT &inter)
{
    LINE ll1, ll2;
    ll1 = makeline(l1.s, l1.e);
    ll2 = makeline(l2.s, l2.e);
    if (lineintersect(ll1, ll2, inter))
        return online(l1, inter) && online(l2, inter);
    else
        return false;
}

/******************************\ 
*         * 
* 多边形常用算法模块    * 
*         * 
\******************************/

// 如果无特别说明，输入多边形顶点要求按逆时针排列

/* 
返回值：输入的多边形是简单多边形，返回true 
要 求：输入顶点序列按逆时针排序 
说 明：简单多边形定义： 
1：循环排序中相邻线段对的交是他们之间共有的单个点 
2：不相邻的线段不相交 
本程序默认第一个条件已经满足 
*/
bool issimple(int vcount, POINT polygon[])
{
    int i, cn;
    LINESEG l1, l2;
    for (i = 0; i < vcount; i++)
    {
        l1.s = polygon[i];
        l1.e = polygon[(i + 1) % vcount];
        cn = vcount - 3;
        while (cn)
        {
            l2.s = polygon[(i + 2) % vcount];
            l2.e = polygon[(i + 3) % vcount];
            if (intersect(l1, l2))
                break;
            cn--;
        }
        if (cn)
            return false;
    }
    return true;
}
// 返回值：按输入顺序返回多边形顶点的凸凹性判断，bc[i]=1,iff:第i个顶点是凸顶点
void checkconvex(int vcount, POINT polygon[], bool bc[])
{
    int i, index = 0;
    POINT tp = polygon[0];
    for (i = 1; i < vcount; i++) // 寻找第一个凸顶点
    {
        if (polygon[i].y < tp.y || (polygon[i].y == tp.y && polygon[i].x < tp.x))
        {
            tp = polygon[i];
            index = i;
        }
    }
    int count = vcount - 1;
    bc[index] = 1;
    while (count) // 判断凸凹性
    {
        if (multiply(polygon[(index + 1) % vcount], polygon[(index + 2) % vcount], polygon[index]) >= 0)
            bc[(index + 1) % vcount] = 1;
        else
            bc[(index + 1) % vcount] = 0;
        index++;
        count--;
    }
}
// 返回值：多边形polygon是凸多边形时，返回true
bool isconvex(int vcount, POINT polygon[])
{
    bool bc[MAXV];
    checkconvex(vcount, polygon, bc);
    for (int i = 0; i < vcount; i++) // 逐一检查顶点，是否全部是凸顶点
        if (!bc[i])
            return false;
    return true;
}
// 返回多边形面积(signed)；输入顶点按逆时针排列时，返回正值；否则返回负值
double area_of_polygon(int vcount, POINT polygon[])
{
    int i;
    double s;
    if (vcount < 3)
        return 0;
    s = polygon[0].y * (polygon[vcount - 1].x - polygon[1].x);
    for (i = 1; i < vcount; i++)
        s += polygon[i].y * (polygon[(i - 1)].x - polygon[(i + 1) % vcount].x);
    return s / 2;
}
// 如果输入顶点按逆时针排列，返回true
bool isconterclock(int vcount, POINT polygon[])
{
    return area_of_polygon(vcount, polygon) > 0;
}
// 另一种判断多边形顶点排列方向的方法
bool isccwize(int vcount, POINT polygon[])
{
    int i, index;
    POINT a, b, v;
    v = polygon[0];
    index = 0;
    for (i = 1; i < vcount; i++) // 找到最低且最左顶点，肯定是凸顶点
    {
        if (polygon[i].y < v.y || polygon[i].y == v.y && polygon[i].x < v.x)
        {
            index = i;
        }
    }
    a = polygon[(index - 1 + vcount) % vcount]; // 顶点v的前一顶点
    b = polygon[(index + 1) % vcount];          // 顶点v的后一顶点
    return multiply(v, b, a) > 0;
}
/********************************************************************************************   
射线法判断点q与多边形polygon的位置关系，要求polygon为简单多边形，顶点逆时针排列 
   如果点在多边形内：   返回0 
   如果点在多边形边上： 返回1 
   如果点在多边形外： 返回2 
*********************************************************************************************/
int insidepolygon(int vcount, POINT Polygon[], POINT q)
{
    int c = 0, i, n;
    LINESEG l1, l2;
    bool bintersect_a, bonline1, bonline2, bonline3;
    double r1, r2;

    l1.s = q;
    l1.e = q;
    l1.e.x = double(INF);
    n = vcount;
    for (i = 0; i < vcount; i++)
    {
        l2.s = Polygon[i];
        l2.e = Polygon[(i + 1) % n];
        if (online(l2, q))
            return 1;                                             // 如果点在边上，返回1
        if ((bintersect_a = intersect_A(l1, l2)) ||               // 相交且不在端点
            ((bonline1 = online(l1, Polygon[(i + 1) % n])) &&     // 第二个端点在射线上
             ((!(bonline2 = online(l1, Polygon[(i + 2) % n]))) && /* 前一个端点和后一个端点在射线两侧 */
                  ((r1 = multiply(Polygon[i], Polygon[(i + 1) % n], l1.s) * multiply(Polygon[(i + 1) % n], Polygon[(i + 2) % n], l1.s)) > 0) ||
              (bonline3 = online(l1, Polygon[(i + 2) % n])) && /* 下一条边是水平线，前一个端点和后一个端点在射线两侧  */
                  ((r2 = multiply(Polygon[i], Polygon[(i + 2) % n], l1.s) * multiply(Polygon[(i + 2) % n],
                                                                                     Polygon[(i + 3) % n], l1.s)) > 0))))
            c++;
    }
    if (c % 2 == 1)
        return 0;
    else
        return 2;
}
//点q是凸多边形polygon内时，返回true；注意：多边形polygon一定要是凸多边形
bool InsideConvexPolygon(int vcount, POINT polygon[], POINT q) // 可用于三角形！
{
    POINT p;
    LINESEG l;
    int i;
    p.x = 0;
    p.y = 0;
    for (i = 0; i < vcount; i++) // 寻找一个肯定在多边形polygon内的点p：多边形顶点平均值
    {
        p.x += polygon[i].x;
        p.y += polygon[i].y;
    }
    p.x /= vcount;
    p.y /= vcount;

    for (i = 0; i < vcount; i++)
    {
        l.s = polygon[i];
        l.e = polygon[(i + 1) % vcount];
        if (multiply(p, l.e, l.s) * multiply(q, l.e, l.s) < 0) /* 点p和点q在边l的两侧，说明点q肯定在多边形外 */
            break;
    }
    return (i == vcount);
}
/********************************************** 
寻找凸包的graham 扫描法 
PointSet为输入的点集； 
ch为输出的凸包上的点集，按照逆时针方向排列; 
n为PointSet中的点的数目 
len为输出的凸包上的点的个数 
**********************************************/
void Graham_scan(POINT PointSet[], POINT ch[], int n, int &len)
{
    int i, j, k = 0, top = 2;
    POINT tmp;
    // 选取PointSet中y坐标最小的点PointSet[k]，如果这样的点有多个，则取最左边的一个
    for (i = 1; i < n; i++)
        if (PointSet[i].y < PointSet[k].y || (PointSet[i].y == PointSet[k].y) && (PointSet[i].x < PointSet[k].x))
            k = i;
    tmp = PointSet[0];
    PointSet[0] = PointSet[k];
    PointSet[k] = tmp;          // 现在PointSet中y坐标最小的点在PointSet[0]
    for (i = 1; i < n - 1; i++) /* 对顶点按照相对PointSet[0]的极角从小到大进行排序，极角相同的按照距离PointSet[0]从近到远进行排序 */
    {
        k = i;
        for (j = i + 1; j < n; j++)
            if (multiply(PointSet[j], PointSet[k], PointSet[0]) > 0 ||    // 极角更小
                (multiply(PointSet[j], PointSet[k], PointSet[0]) == 0) && /* 极角相等，距离更短 */
                    dist(PointSet[0], PointSet[j]) < dist(PointSet[0], PointSet[k]))
                k = j;
        tmp = PointSet[i];
        PointSet[i] = PointSet[k];
        PointSet[k] = tmp;
    }
    ch[0] = PointSet[0];
    ch[1] = PointSet[1];
    ch[2] = PointSet[2];
    for (i = 3; i < n; i++)
    {
        while (multiply(PointSet[i], ch[top], ch[top - 1]) >= 0)
            top--;
        ch[++top] = PointSet[i];
    }
    len = top + 1;
}
// 卷包裹法求点集凸壳，参数说明同graham算法
void ConvexClosure(POINT PointSet[], POINT ch[], int n, int &len)
{
    int top = 0, i, index, first;
    double curmax, curcos, curdis;
    POINT tmp;
    LINESEG l1, l2;
    bool use[MAXV];
    tmp = PointSet[0];
    index = 0;
    // 选取y最小点，如果多于一个，则选取最左点
    for (i = 1; i < n; i++)
    {
        if (PointSet[i].y < tmp.y || PointSet[i].y == tmp.y && PointSet[i].x < tmp.x)
        {
            index = i;
        }
        use[i] = false;
    }
    tmp = PointSet[index];
    first = index;
    use[index] = true;

    index = -1;
    ch[top++] = tmp;
    tmp.x -= 100;
    l1.s = tmp;
    l1.e = ch[0];
    l2.s = ch[0];

    while (index != first)
    {
        curmax = -100;
        curdis = 0;
        // 选取与最后一条确定边夹角最小的点，即余弦值最大者
        for (i = 0; i < n; i++)
        {
            if (use[i])
                continue;
            l2.e = PointSet[i];
            curcos = cosine(l1, l2); // 根据cos值求夹角余弦，范围在 （-1 -- 1 ）
            if (curcos > curmax || fabs(curcos - curmax) < 1e-6 && dist(l2.s, l2.e) > curdis)
            {
                curmax = curcos;
                index = i;
                curdis = dist(l2.s, l2.e);
            }
        }
        use[first] = false; //清空第first个顶点标志，使最后能形成封闭的hull
        use[index] = true;
        ch[top++] = PointSet[index];
        l1.s = ch[top - 2];
        l1.e = ch[top - 1];
        l2.s = ch[top - 1];
    }
    len = top - 1;
}
/*********************************************************************************************  
 判断线段是否在简单多边形内(注意：如果多边形是凸多边形，下面的算法可以化简) 
    必要条件一：线段的两个端点都在多边形内； 
 必要条件二：线段和多边形的所有边都不内交； 
 用途： 1. 判断折线是否在简单多边形内 
   2. 判断简单多边形是否在另一个简单多边形内 
**********************************************************************************************/
bool LinesegInsidePolygon(int vcount, POINT polygon[], LINESEG l)
{
    // 判断线端l的端点是否不都在多边形内
    if (!insidepolygon(vcount, polygon, l.s) || !insidepolygon(vcount, polygon, l.e))
        return false;
    int top = 0, i, j;
    POINT PointSet[MAXV], tmp;
    LINESEG s;

    for (i = 0; i < vcount; i++)
    {
        s.s = polygon[i];
        s.e = polygon[(i + 1) % vcount];
        if (online(s, l.s)) //线段l的起始端点在线段s上
            PointSet[top++] = l.s;
        else if (online(s, l.e)) //线段l的终止端点在线段s上
            PointSet[top++] = l.e;
        else
        {
            if (online(l, s.s)) //线段s的起始端点在线段l上
                PointSet[top++] = s.s;
            else if (online(l, s.e)) // 线段s的终止端点在线段l上
                PointSet[top++] = s.e;
            else
            {
                if (intersect(l, s)) // 这个时候如果相交，肯定是内交，返回false
                    return false;
            }
        }
    }

    for (i = 0; i < top - 1; i++) /* 冒泡排序，x坐标小的排在前面；x坐标相同者，y坐标小的排在前面 */
    {
        for (j = i + 1; j < top; j++)
        {
            if (PointSet[i].x > PointSet[j].x || fabs(PointSet[i].x - PointSet[j].x) < EP && PointSet[i].y > PointSet[j].y)
            {
                tmp = PointSet[i];
                PointSet[i] = PointSet[j];
                PointSet[j] = tmp;
            }
        }
    }

    for (i = 0; i < top - 1; i++)
    {
        tmp.x = (PointSet[i].x + PointSet[i + 1].x) / 2; //得到两个相邻交点的中点
        tmp.y = (PointSet[i].y + PointSet[i + 1].y) / 2;
        if (!insidepolygon(vcount, polygon, tmp))
            return false;
    }
    return true;
}
/*********************************************************************************************  
求任意简单多边形polygon的重心 
需要调用下面几个函数： 
 void AddPosPart(); 增加右边区域的面积 
 void AddNegPart(); 增加左边区域的面积 
 void AddRegion(); 增加区域面积 
在使用该程序时，如果把xtr,ytr,wtr,xtl,ytl,wtl设成全局变量就可以使这些函数的形式得到化简,
但要注意函数的声明和调用要做相应变化 
**********************************************************************************************/
void AddPosPart(double x, double y, double w, double &xtr, double &ytr, double &wtr)
{
    if (abs(wtr + w) < 1e-10)
        return; // detect zero regions
    xtr = (wtr * xtr + w * x) / (wtr + w);
    ytr = (wtr * ytr + w * y) / (wtr + w);
    wtr = w + wtr;
    return;
}
void AddNegPart(double x, ouble y, double w, double &xtl, double &ytl, double &wtl)
{
    if (abs(wtl + w) < 1e-10)
        return; // detect zero regions

    xtl = (wtl * xtl + w * x) / (wtl + w);
    ytl = (wtl * ytl + w * y) / (wtl + w);
    wtl = w + wtl;
    return;
}
void AddRegion(double x1, double y1, double x2, double y2, double &xtr, double &ytr,
               double &wtr, double &xtl, double &ytl, double &wtl)
{
    if (abs(x1 - x2) < 1e-10)
        return;

    if (x2 > x1)
    {
        AddPosPart((x2 + x1) / 2, y1 / 2, (x2 - x1) * y1, xtr, ytr, wtr); /* rectangle 全局变量变化处 */
        AddPosPart((x1 + x2 + x2) / 3, (y1 + y1 + y2) / 3, (x2 - x1) * (y2 - y1) / 2, xtr, ytr, wtr);
        // triangle 全局变量变化处
    }
    else
    {
        AddNegPart((x2 + x1) / 2, y1 / 2, (x2 - x1) * y1, xtl, ytl, wtl);
        // rectangle 全局变量变化处
        AddNegPart((x1 + x2 + x2) / 3, (y1 + y1 + y2) / 3, (x2 - x1) * (y2 - y1) / 2, xtl, ytl, wtl);
        // triangle  全局变量变化处
    }
}
POINT cg_simple(int vcount, POINT polygon[])
{
    double xtr, ytr, wtr, xtl, ytl, wtl;
    //注意： 如果把xtr,ytr,wtr,xtl,ytl,wtl改成全局变量后这里要删去
    POINT p1, p2, tp;
    xtr = ytr = wtr = 0.0;
    xtl = ytl = wtl = 0.0;
    for (int i = 0; i < vcount; i++)
    {
        p1 = polygon[i];
        p2 = polygon[(i + 1) % vcount];
        AddRegion(p1.x, p1.y, p2.x, p2.y, xtr, ytr, wtr, xtl, ytl, wtl); //全局变量变化处
    }
    tp.x = (wtr * xtr + wtl * xtl) / (wtr + wtl);
    tp.y = (wtr * ytr + wtl * ytl) / (wtr + wtl);
    return tp;
}
// 求凸多边形的重心,要求输入多边形按逆时针排序
POINT gravitycenter(int vcount, POINT polygon[])
{
    POINT tp;
    double x, y, s, x0, y0, cs, k;
    x = 0;
    y = 0;
    s = 0;
    for (int i = 1; i < vcount - 1; i++)
    {
        x0 = (polygon[0].x + polygon[i].x + polygon[i + 1].x) / 3;
        y0 = (polygon[0].y + polygon[i].y + polygon[i + 1].y) / 3; //求当前三角形的重心
        cs = multiply(polygon[i], polygon[i + 1], polygon[0]) / 2;
        //三角形面积可以直接利用该公式求解
        if (abs(s) < 1e-20)
        {
            x = x0;
            y = y0;
            s += cs;
            continue;
        }
        k = cs / s; //求面积比例
        x = (x + k * x0) / (1 + k);
        y = (y + k * y0) / (1 + k);
        s += cs;
    }
    tp.x = x;
    tp.y = y;
    return tp;
}

/************************************************
给定一简单多边形，找出一个肯定在该多边形内的点 
定理1 ：每个多边形至少有一个凸顶点 
定理2 ：顶点数>=4的简单多边形至少有一条对角线 
结论 ： x坐标最大，最小的点肯定是凸顶点 
 y坐标最大，最小的点肯定是凸顶点            
************************************************/
POINT a_point_insidepoly(int vcount, POINT polygon[])
{
    POINT v, a, b, r;
    int i, index;
    v = polygon[0];
    index = 0;
    for (i = 1; i < vcount; i++) //寻找一个凸顶点
    {
        if (polygon[i].y < v.y)
        {
            v = polygon[i];
            index = i;
        }
    }
    a = polygon[(index - 1 + vcount) % vcount]; //得到v的前一个顶点
    b = polygon[(index + 1) % vcount];          //得到v的后一个顶点
    POINT tri[3], q;
    tri[0] = a;
    tri[1] = v;
    tri[2] = b;
    double md = INF;
    int in1 = index;
    bool bin = false;
    for (i = 0; i < vcount; i++) //寻找在三角形avb内且离顶点v最近的顶点q
    {
        if (i == index)
            continue;
        if (i == (index - 1 + vcount) % vcount)
            continue;
        if (i == (index + 1) % vcount)
            continue;
        if (!InsideConvexPolygon(3, tri, polygon[i]))
            continue;
        bin = true;
        if (dist(v, polygon[i]) < md)
        {
            q = polygon[i];
            md = dist(v, q);
        }
    }
    if (!bin) //没有顶点在三角形avb内，返回线段ab中点
    {
        r.x = (a.x + b.x) / 2;
        r.y = (a.y + b.y) / 2;
        return r;
    }
    r.x = (v.x + q.x) / 2; //返回线段vq的中点
    r.y = (v.y + q.y) / 2;
    return r;
}
/***********************************************************************************************
求从多边形外一点p出发到一个简单多边形的切线,如果存在返回切点,其中rp点是右切点,lp是左切点 
注意：p点一定要在多边形外 ,输入顶点序列是逆时针排列 
原 理： 如果点在多边形内肯定无切线;凸多边形有唯一的两个切点,凹多边形就可能有多于两个的切点; 
  如果polygon是凸多边形，切点只有两个只要找到就可以,可以化简此算法 
  如果是凹多边形还有一种算法可以求解:先求凹多边形的凸包,然后求凸包的切线 
/***********************************************************************************************/
void pointtangentpoly(int vcount, POINT polygon[], POINT p, POINT &rp, POINT &lp)
{
    LINESEG ep, en;
    bool blp, bln;
    rp = polygon[0];
    lp = polygon[0];
    for (int i = 1; i < vcount; i++)
    {
        ep.s = polygon[(i + vcount - 1) % vcount];
        ep.e = polygon[i];
        en.s = polygon[i];
        en.e = polygon[(i + 1) % vcount];
        blp = multiply(ep.e, p, ep.s) >= 0; // p is to the left of pre edge
        bln = multiply(en.e, p, en.s) >= 0; // p is to the left of next edge
        if (!blp && bln)
        {
            if (multiply(polygon[i], rp, p) > 0) // polygon[i] is above rp
                rp = polygon[i];
        }
        if (blp && !bln)
        {
            if (multiply(lp, polygon[i], p) > 0) // polygon[i] is below lp
                lp = polygon[i];
        }
    }
    return;
}
// 如果多边形polygon的核存在，返回true，返回核上的一点p.顶点按逆时针方向输入
bool core_exist(int vcount, POINT polygon[], POINT &p)
{
    int i, j, k;
    LINESEG l;
    LINE lineset[MAXV];
    for (i = 0; i < vcount; i++)
    {
        lineset[i] = makeline(polygon[i], polygon[(i + 1) % vcount]);
    }
    for (i = 0; i < vcount; i++)
    {
        for (j = 0; j < vcount; j++)
        {
            if (i == j)
                continue;
            if (lineintersect(lineset[i], lineset[j], p))
            {
                for (k = 0; k < vcount; k++)
                {
                    l.s = polygon[k];
                    l.e = polygon[(k + 1) % vcount];
                    if (multiply(p, l.e, l.s) > 0)
                        //多边形顶点按逆时针方向排列，核肯定在每条边的左侧或边上
                        break;
                }
                if (k == vcount) //找到了一个核上的点
                    break;
            }
        }
        if (j < vcount)
            break;
    }
    if (i < vcount)
        return true;
    else
        return false;
}
/*************************\ 
*       * 
* 圆的基本运算           * 
*          * 
\*************************/
/******************************************************************************
返回值 ： 点p在圆内(包括边界)时，返回true 
用途 ： 因为圆为凸集，所以判断点集，折线，多边形是否在圆内时，
 只需要逐一判断点是否在圆内即可。 
*******************************************************************************/
bool point_in_circle(POINT o, double r, POINT p)
{
    double d2 = (p.x - o.x) * (p.x - o.x) + (p.y - o.y) * (p.y - o.y);
    double r2 = r * r;
    return d2 < r2 || abs(d2 - r2) < EP;
}
/******************************************************************************
用 途 ：求不共线的三点确定一个圆 
输 入 ：三个点p1,p2,p3 
返回值 ：如果三点共线，返回false；反之，返回true。圆心由q返回，半径由r返回 
*******************************************************************************/
bool cocircle(POINT p1, POINT p2, POINT p3, POINT &q, double &r)
{
    double x12 = p2.x - p1.x;
    double y12 = p2.y - p1.y;
    double x13 = p3.x - p1.x;
    double y13 = p3.y - p1.y;
    double z2 = x12 * (p1.x + p2.x) + y12 * (p1.y + p2.y);
    double z3 = x13 * (p1.x + p3.x) + y13 * (p1.y + p3.y);
    double d = 2.0 * (x12 * (p3.y - p2.y) - y12 * (p3.x - p2.x));
    if (abs(d) < EP) //共线，圆不存在
        return false;
    q.x = (y13 * z2 - y12 * z3) / d;
    q.y = (x12 * z3 - x13 * z2) / d;
    r = dist(p1, q);
    return true;
}
int line_circle(LINE l, POINT o, double r, POINT &p1, POINT &p2)
{
    return true;
}

/**************************\ 
*        * 
* 矩形的基本运算          * 
*                         * 
\**************************/
/* 
说明：因为矩形的特殊性，常用算法可以化简： 
1.判断矩形是否包含点 
只要判断该点的横坐标和纵坐标是否夹在矩形的左右边和上下边之间。 
2.判断线段、折线、多边形是否在矩形中 
因为矩形是个凸集，所以只要判断所有端点是否都在矩形中就可以了。 
3.判断圆是否在矩形中 
圆在矩形中的充要条件是：圆心在矩形中且圆的半径小于等于圆心到矩形四边的距离的最小值。 
*/
// 已知矩形的三个顶点(a,b,c)，计算第四个顶点d的坐标. 注意：已知的三个顶点可以是无序的
POINT rect4th(POINT a, POINT b, POINT c)
{
    POINT d;
    if (abs(dotmultiply(a, b, c)) < EP) // 说明c点是直角拐角处
    {
        d.x = a.x + b.x - c.x;
        d.y = a.y + b.y - c.y;
    }
    if (abs(dotmultiply(a, c, b)) < EP) // 说明b点是直角拐角处
    {
        d.x = a.x + c.x - b.x;
        d.y = a.y + c.y - b.x;
    }
    if (abs(dotmultiply(c, b, a)) < EP) // 说明a点是直角拐角处
    {
        d.x = c.x + b.x - a.x;
        d.y = c.y + b.y - a.y;
    }
    return d;
}

/*************************\ 
*      * 
* 常用算法的描述  * 
*      * 
\*************************/
/* 
尚未实现的算法： 
1. 求包含点集的最小圆 
2. 求多边形的交 
3. 简单多边形的三角剖分 
4. 寻找包含点集的最小矩形 
5. 折线的化简 
6. 判断矩形是否在矩形中 
7. 判断矩形能否放在矩形中 
8. 矩形并的面积与周长 
9. 矩形并的轮廓 
10.矩形并的闭包 
11.矩形的交 
12.点集中的最近点对 
13.多边形的并 
14.圆的交与并 
15.直线与圆的关系 
16.线段与圆的关系 
17.求多边形的核监视摄象机 
18.求点集中不相交点对 railwai 
*/
/* 
寻找包含点集的最小矩形 
原理：该矩形至少一条边与点集的凸壳的某条边共线 
First take the convex hull of the points. Let the resulting convex 
polygon be P. It has been known for some time that the minimum 
area rectangle enclosing P must have one rectangle side flush with 
(i.e., collinear with and overlapping) one edge of P. This geometric 
fact was used by Godfried Toussaint to develop the "rotating calipers" 
algorithm in a hard-to-find 1983 paper, "Solving Geometric Problems 
with the Rotating Calipers" (Proc. IEEE MELECON). The algorithm 
rotates a surrounding rectangle from one flush edge to the next, 
keeping track of the minimum area for each edge. It achieves O(n) 
time (after hull computation). See the "Rotating Calipers Homepage" 
http://www.cs.mcgill.ca/~orm/rotcal.frame.html for a description 
and applet. 
*/
/* 
折线的化简 伪码如下： 
Input: tol = the approximation tolerance 
L = {V0,V1,,Vn-1} is any n-vertex polyline 

Set start = 0; 
Set k = 0; 
Set W0 = V0; 
for each vertex Vi (i=1,n-1) 
{ 
if Vi is within tol from Vstart 
then ignore it, and continue with the next vertex 

Vi is further than tol away from Vstart 
so add it as a new vertex of the reduced polyline 
Increment k++; 
Set Wk = Vi; 
Set start = i; as the new initial vertex 
} 

Output: W = {W0,W1,,Wk-1} = the k-vertex simplified polyline 
*/
/********************\ 
*        * 
* 补充    * 
*     * 
\********************/

//两圆关系：
/* 两圆： 
相离： return 1； 
外切： return 2； 
相交： return 3； 
内切： return 4； 
内含： return 5； 
*/
int CircleRelation(POINT p1, double r1, POINT p2, double r2)
{
    double d = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));

    if (fabs(d - r1 - r2) < EP) // 必须保证前两个if先被判定！
        return 2;
    if (fabs(d - fabs(r1 - r2)) < EP)
        return 4;
    if (d > r1 + r2)
        return 1;
    if (d < fabs(r1 - r2))
        return 5;
    if (fabs(r1 - r2) < d && d < r1 + r2)
        return 3;
    return 0; // indicate an error!
}
//判断圆是否在矩形内：
// 判定圆是否在矩形内，是就返回true（设矩形水平，且其四个顶点由左上开始按顺时针排列）
// 调用ptoldist函数，在第4页
bool CircleRecRelation(POINT pc, double r, POINT pr1, POINT pr2, POINT pr3, POINT pr4)
{
    if (pr1.x < pc.x && pc.x < pr2.x && pr3.y < pc.y && pc.y < pr2.y)
    {
        LINESEG line1(pr1, pr2);
        LINESEG line2(pr2, pr3);
        LINESEG line3(pr3, pr4);
        LINESEG line4(pr4, pr1);
        if (r < ptoldist(pc, line1) && r < ptoldist(pc, line2) && r < ptoldist(pc, line3) && r < ptoldist(pc, line4))
            return true;
    }
    return false;
}
//点到平面的距离：
//点到平面的距离,平面用一般式表示ax+by+cz+d=0
double P2planeDist(double x, double y, double z, double a, double b, double c, double d)
{
    return fabs(a * x + b * y + c * z + d) / sqrt(a * a + b * b + c * c);
}
//点是否在直线同侧：
//两个点是否在直线同侧，是则返回true
bool SameSide(POINT p1, POINT p2, LINE line)
{
    return (line.a * p1.x + line.b * p1.y + line.c) *
               (line.a * p2.x + line.b * p2.y + line.c) >
           0;
}
//镜面反射线：
// 已知入射线、镜面，求反射线。
// a1,b1,c1为镜面直线方程(a1 x + b1 y + c1 = 0 ,下同)系数;
//a2,b2,c2为入射光直线方程系数;
//a,b,c为反射光直线方程系数.
// 光是有方向的，使用时注意：入射光向量:<-b2,a2>；反射光向量:<b,-a>.
// 不要忘记结果中可能会有"negative zeros"
void reflect(double a1, double b1, double c1, double a2, double b2, double c2, double &a, double &b, double &c)
{
    double n, m;
    double tpb, tpa;
    tpb = b1 * b2 + a1 * a2;
    tpa = a2 * b1 - a1 * b2;
    m = (tpb * b1 + tpa * a1) / (b1 * b1 + a1 * a1);
    n = (tpa * b1 - tpb * a1) / (b1 * b1 + a1 * a1);
    if (fabs(a1 * b2 - a2 * b1) < 1e-20)
    {
        a = a2;
        b = b2;
        c = c2;
        return;
    }
    double xx, yy; //(xx,yy)是入射线与镜面的交点。
    xx = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1);
    yy = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1);
    a = n;
    b = -m;
    c = m * yy - xx * n;
}
//矩形包含：
// 矩形2（C，D）是否在1（A，B）内
bool r2inr1(double A, double B, double C, double D)
{
    double X, Y, L, K, DMax;
    if (A < B)
    {
        double tmp = A;
        A = B;
        B = tmp;
    }
    if (C < D)
    {
        double tmp = C;
        C = D;
        D = tmp;
    }
    if (A > C && B > D) // trivial case
        return true;
    else if (D >= B)
        return false;
    else
    {
        X = sqrt(A * A + B * B); // outer rectangle's diagonal
        Y = sqrt(C * C + D * D); // inner rectangle's diagonal
        if (Y < B)               // check for marginal conditions
            return true;         // the inner rectangle can freely rotate inside
        else if (Y > X)
            return false;
        else
        {
            L = (B - sqrt(Y * Y - A * A)) / 2;
            K = (A - sqrt(Y * Y - B * B)) / 2;
            DMax = sqrt(L * L + K * K);
            if (D >= DMax)
                return false;
            else
                return true;
        }
    }
}
//两圆交点：
// 两圆已经相交（相切）
void c2point(POINT p1, double r1, POINT p2, double r2, POINT &rp1, POINT &rp2)
{
    double a, b, r;
    a = p2.x - p1.x;
    b = p2.y - p1.y;
    r = (a * a + b * b + r1 * r1 - r2 * r2) / 2;
    if (a == 0 && b != 0)
    {
        rp1.y = rp2.y = r / b;
        rp1.x = sqrt(r1 * r1 - rp1.y * rp1.y);
        rp2.x = -rp1.x;
    }
    else if (a != 0 && b == 0)
    {
        rp1.x = rp2.x = r / a;
        rp1.y = sqrt(r1 * r1 - rp1.x * rp2.x);
        rp2.y = -rp1.y;
    }
    else if (a != 0 && b != 0)
    {
        double delta;
        delta = b * b * r * r - (a * a + b * b) * (r * r - r1 * r1 * a * a);
        rp1.y = (b * r + sqrt(delta)) / (a * a + b * b);
        rp2.y = (b * r - sqrt(delta)) / (a * a + b * b);
        rp1.x = (r - b * rp1.y) / a;
        rp2.x = (r - b * rp2.y) / a;
    }

    rp1.x += p1.x;
    rp1.y += p1.y;
    rp2.x += p1.x;
    rp2.y += p1.y;
}
//两圆公共面积：
// 必须保证相交
double c2area(POINT p1, double r1, POINT p2, double r2)
{
    POINT rp1, rp2;
    c2point(p1, r1, p2, r2, rp1, rp2);

    if (r1 > r2) //保证r2>r1
    {
        swap(p1, p2);
        swap(r1, r2);
    }
    double a, b, rr;
    a = p1.x - p2.x;
    b = p1.y - p2.y;
    rr = sqrt(a * a + b * b);

    double dx1, dy1, dx2, dy2;
    double sita1, sita2;
    dx1 = rp1.x - p1.x;
    dy1 = rp1.y - p1.y;
    dx2 = rp2.x - p1.x;
    dy2 = rp2.y - p1.y;
    sita1 = acos((dx1 * dx2 + dy1 * dy2) / r1 / r1);

    dx1 = rp1.x - p2.x;
    dy1 = rp1.y - p2.y;
    dx2 = rp2.x - p2.x;
    dy2 = rp2.y - p2.y;
    sita2 = acos((dx1 * dx2 + dy1 * dy2) / r2 / r2);
    double s = 0;
    if (rr < r2) //相交弧为优弧
        s = r1 * r1 * (PI - sita1 / 2 + sin(sita1) / 2) + r2 * r2 * (sita2 - sin(sita2)) / 2;
    else //相交弧为劣弧
        s = (r1 * r1 * (sita1 - sin(sita1)) + r2 * r2 * (sita2 - sin(sita2))) / 2;

    return s;
}
//圆和直线关系：
//0----相离 1----相切 2----相交
int clpoint(POINT p, double r, double a, double b, double c, POINT &rp1, POINT &rp2)
{
    int res = 0;

    c = c + a * p.x + b * p.y;
    double tmp;
    if (a == 0 && b != 0)
    {
        tmp = -c / b;
        if (r * r < tmp * tmp)
            res = 0;
        else if (r * r == tmp * tmp)
        {
            res = 1;
            rp1.y = tmp;
            rp1.x = 0;
        }
        else
        {
            res = 2;
            rp1.y = rp2.y = tmp;
            rp1.x = sqrt(r * r - tmp * tmp);
            rp2.x = -rp1.x;
        }
    }
    else if (a != 0 && b == 0)
    {
        tmp = -c / a;
        if (r * r < tmp * tmp)
            res = 0;
        else if (r * r == tmp * tmp)
        {
            res = 1;
            rp1.x = tmp;
            rp1.y = 0;
        }
        else
        {
            res = 2;
            rp1.x = rp2.x = tmp;
            rp1.y = sqrt(r * r - tmp * tmp);
            rp2.y = -rp1.y;
        }
    }
    else if (a != 0 && b != 0)
    {
        double delta;
        delta = b * b * c * c - (a * a + b * b) * (c * c - a * a * r * r);
        if (delta < 0)
            res = 0;
        else if (delta == 0)
        {
            res = 1;
            rp1.y = -b * c / (a * a + b * b);
            rp1.x = (-c - b * rp1.y) / a;
        }
        else
        {
            res = 2;
            rp1.y = (-b * c + sqrt(delta)) / (a * a + b * b);
            rp2.y = (-b * c - sqrt(delta)) / (a * a + b * b);
            rp1.x = (-c - b * rp1.y) / a;
            rp2.x = (-c - b * rp2.y) / a;
        }
    }
    rp1.x += p.x;
    rp1.y += p.y;
    rp2.x += p.x;
    rp2.y += p.y;
    return res;
}
//内切圆：
void incircle(POINT p1, POINT p2, POINT p3, POINT &rp, double &r)
{
    double dx31, dy31, dx21, dy21, d31, d21, a1, b1, c1;
    dx31 = p3.x - p1.x;
    dy31 = p3.y - p1.y;
    dx21 = p2.x - p1.x;
    dy21 = p2.y - p1.y;

    d31 = sqrt(dx31 * dx31 + dy31 * dy31);
    d21 = sqrt(dx21 * dx21 + dy21 * dy21);
    a1 = dx31 * d21 - dx21 * d31;
    b1 = dy31 * d21 - dy21 * d31;
    c1 = a1 * p1.x + b1 * p1.y;

    double dx32, dy32, dx12, dy12, d32, d12, a2, b2, c2;
    dx32 = p3.x - p2.x;
    dy32 = p3.y - p2.y;
    dx12 = -dx21;
    dy12 = -dy21;

    d32 = sqrt(dx32 * dx32 + dy32 * dy32);
    d12 = d21;
    a2 = dx12 * d32 - dx32 * d12;
    b2 = dy12 * d32 - dy32 * d12;
    c2 = a2 * p2.x + b2 * p2.y;

    rp.x = (c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1);
    rp.y = (c2 * a1 - c1 * a2) / (a1 * b2 - a2 * b1);
    r = fabs(dy21 * rp.x - dx21 * rp.y + dx21 * p1.y - dy21 * p1.x) / d21;
}
//求切点：
// p---圆心坐标， r---圆半径， sp---圆外一点， rp1,rp2---切点坐标
void cutpoint(POINT p, double r, POINT sp, POINT &rp1, POINT &rp2)
{
    POINT p2;
    p2.x = (p.x + sp.x) / 2;
    p2.y = (p.y + sp.y) / 2;

    double dx2, dy2, r2;
    dx2 = p2.x - p.x;
    dy2 = p2.y - p.y;
    r2 = sqrt(dx2 * dx2 + dy2 * dy2);
    c2point(p, r, p2, r2, rp1, rp2);
}
//线段的左右旋：
/* l2在l1的左/右方向（l1为基准线） 
返回 0 ： 重合； 
返回 1 ： 右旋； 
返回 –1 ： 左旋； 
*/
int rotat(LINESEG l1, LINESEG l2)
{
    double dx1, dx2, dy1, dy2;
    dx1 = l1.s.x - l1.e.x;
    dy1 = l1.s.y - l1.e.y;
    dx2 = l2.s.x - l2.e.x;
    dy2 = l2.s.y - l2.e.y;

    double d;
    d = dx1 * dy2 - dx2 * dy1;
    if (d == 0)
        return 0;
    else if (d > 0)
        return -1;
    else
        return 1;
}

/*
公式： 

球坐标公式： 
直角坐标为 P(x, y, z) 时，对应的球坐标是(rsinφcosθ, rsinφsinθ, rcosφ),其中φ是向量OP与Z轴的夹角，范围[0，π]；是OP在XOY面上的投影到X轴的旋角，范围[0，2π]  

直线的一般方程转化成向量方程： 
ax+by+c=0 
x-x0     y-y0 
   ------ = ------- // (x0,y0)为直线上一点，m,n为向量 
m        n 
转换关系： 
a=n；b=-m；c=m·y0-n·x0； 
m=-b; n=a; 

三点平面方程： 
三点为P1，P2，P3 
设向量  M1=P2-P1; M2=P3-P1; 
平面法向量：  M=M1 x M2 （） 
平面方程：    M.i(x-P1.x)+M.j(y-P1.y)+M.k(z-P1.z)=0
*/
```

###### 11.2矩阵面积交

```C++
//  Created by TaoSama on 2015-10-04
//  Copyright (c) 2015 TaoSama. All rights reserved.
//
//#pragma comment(linker, "/STACK:1024000000,1024000000")
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <set>
#include <vector>

using namespace std;
#define pr(x) cout << #x << " = " << x << "  "
#define prln(x) cout << #x << " = " << x << endl
const int N = 2e3 + 10, INF = 0x3f3f3f3f, MOD = 1e9 + 7;

int n;
struct Seg {
    double l, r, h; int d;
    Seg() {}
    Seg(double l, double r, double h, double d): l(l), r(r), h(h), d(d) {}
    bool operator< (const Seg& rhs) const {
        return h < rhs.h;
    }
} a[N];

int cnt[N << 2];
double one[N << 2], two[N << 2], all[N];

#define lson l, m, rt << 1
#define rson m + 1, r, rt << 1 | 1

void push_up(int l, int r, int rt) {
    if(cnt[rt] >= 2) two[rt] = one[rt] = all[r + 1] - all[l];
    else if(cnt[rt] == 1) {
        one[rt] = all[r + 1] - all[l];
        if(l == r) two[rt] = 0;
        else two[rt] = one[rt << 1] + one[rt << 1 | 1];
    } else {
        if(l == r) one[rt] = two[rt] = 0;
        else {
            one[rt] = one[rt << 1] + one[rt << 1 | 1];
            two[rt] = two[rt << 1] + two[rt << 1 | 1];
        }
    }
}

void update(int L, int R, int v, int l, int r, int rt) {
    if(L <= l && r <= R) {
        cnt[rt] += v;
        push_up(l, r, rt);
        return;
    }
    int m = l + r >> 1;
    if(L <= m) update(L, R, v, lson);
    if(R > m) update(L, R, v, rson);
    push_up(l, r, rt);
}

int main() {
//#ifdef LOCAL
    freopen("data.txt", "r", stdin);
//  freopen("out.txt","w",stdout);
//#endif
    ios_base::sync_with_stdio(0);

    int t; scanf("%d", &t);
    while(t--) {
        scanf("%d", &n);
        for(int i = 1; i <= n; ++i) {
            double x1, y1, x2, y2;
            scanf("%lf%lf%lf%lf", &x1, &y1, &x2, &y2);
            a[i] = Seg(x1, x2, y1, 1);
            a[i + n] = Seg(x1, x2, y2, -1);
            all[i] = x1; all[i + n] = x2;
        }
        n <<= 1;
        sort(a + 1, a + 1 + n);
        sort(all + 1, all + 1 + n);
        int m = unique(all + 1, all + 1 + n) - all - 1;

        memset(cnt, 0, sizeof cnt);
        memset(one, 0, sizeof one);
        memset(two, 0, sizeof two);

        double ans = 0;
        for(int i = 1; i < n; ++i) {
            int l = lower_bound(all + 1, all + 1 + m, a[i].l) - all;
            int r = lower_bound(all + 1, all + 1 + m, a[i].r) - all;
            if(l < r) update(l, r - 1, a[i].d, 1, m, 1);
            ans += two[1] * (a[i + 1].h - a[i].h);
        }
        printf("%.2f\n", ans);
    }
    return 0;
}
```

###### 11.2矩阵面积并

```C++
#include<cstdio>
#include<algorithm>
#include<cstring>
#include<iostream>
using namespace std;
typedef long long ll;
const int N=10010+5;
struct Node//矩形
{
    ll x1,y1,x2,y2;
}nodes[N];


ll x[N],y[N];
bool mp[N][N];

int findd(ll *x,ll val,ll n)//在数组x中找到val值的位置
{
    int L=0,R=n-1;
    while(R>=L)
    {
        int mid=L+(R-L)/2;
        if(x[mid]==val) return mid;
        else if(x[mid]>val) R=mid-1;
        else L=mid+1;
    }
    return -1;
}


int main()
{
//    freopen("data.txt","r",stdin);
    ll n,num1,num2;
    while(~scanf("%lld",&n))
    {
        if(n==0) break;
        num1=num2=0;//num1记录有多少个不同x值,num2记录y的
        memset(mp,0,sizeof(mp));
        for(int i=0;i<n;++i)
        {
            scanf("%lld%lld%lld%lld",&nodes[i].x1,&nodes[i].y1,&nodes[i].x2,&nodes[i].y2);
            x[num1++]=nodes[i].x1;
            x[num1++]=nodes[i].x2;
            y[num2++]=nodes[i].y1;
            y[num2++]=nodes[i].y2;
        }
        sort(x,x+num1);
        sort(y,y+num2);
        num1=unique(x,x+num1)-x;//去重
        num2=unique(y,y+num2)-y;//去重

        for(int i=0;i<n;++i)
        {
            int L_x=findd(x,nodes[i].x1,num1);
            int R_x=findd(x,nodes[i].x2,num1);
            int L_y=findd(y,nodes[i].y1,num2);
            int R_y=findd(y,nodes[i].y2,num2);

            for(int j=L_x;j<R_x;++j)
            for(int k=L_y;k<R_y;++k)
                mp[j][k]=true;
        }
        ll ans=0;
        for(int i=0;i<num1;++i)
        for(int j=0;j<num2;++j)if(mp[i][j])
            ans += (x[i+1]-x[i])*(y[j+1]-y[j]);
        printf("%lld\n",ans);
    }
    printf("*\n");
    return 0;
}
```

