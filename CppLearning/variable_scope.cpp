#include <iostream>

// 全局变量，具有外部链接，默认存储类为extern
int globalVar;

void function() {
    // 局部变量，具有自动存储期，默认存储类为auto
    // auto int localVar = 10;会显示报错，两个类型说明符同时出现就是 "two or
    // more data types in declaration"。
    int localVar = 10;

    // 静态变量，具有静态存储期，生命周期贯穿整个程序
    static int staticVar = 20;

    const int constVar = 30; // const变量默认具有static存储期

    // 尝试修改const变量，编译错误
    // constVar = 40;

    // mutable成员变量，可以在const成员函数中修改
    class MyClass {
      public:
	mutable int mutableVar;

	void constMemberFunc() const {
	    mutableVar = 50; // 允许修改mutable成员变量
	}
    };

    // 线程局部变量，每个线程有自己的独立副本
    thread_local int threadVar = 60;
}

int main() {
    extern int externalVar; // 声明具有外部链接的变量

    function();

    return 0;
}