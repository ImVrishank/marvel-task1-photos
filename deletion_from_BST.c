#include<conio.h>
#include<stdio.h>

struct node{
    int data;
    struct node* lchild;
    struct node* rchild;
};

struct node* findmin(struct node* root){
    while(root->lchild != NULL){
        root = root->lchild;
    }
    return root;
}

struct node* deletion(struct node* root, int data){
    struct node* temp;
    if(root == NULL){
        return root;
    }
    else if(data > root->data){
        root->rchild = deletion(root->rchild, data);
    }
    else if(data < root->data){
        root->lchild = deletion(root->lchild, data);
    }
    else{
        // no child
        if(root->lchild == NULL && root->rchild == NULL){
            free(root);
            return NULL;
        }

        // one child
        if(root->lchild == NULL){
            struct node* temp = root->rchild;
            free(root);
            return temp;
        }
        else if(root->rchild == NULL){
            temp = root->lchild;
            free(root);
            return temp;
        }

        // two children
        temp = findmin(root->rchild);
        root->data = temp->data;
        root->right = deletion(root->right; temp->data);
    }

    return root;
}   
