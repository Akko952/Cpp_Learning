#include <iostream>
#include <string>

using namespace std;

// 声明一个结构体类型 Books
struct Books {
    string title;
    string author;
    string subject;
    int book_id;

    // 构造函数
    Books(string t, string a, string s, int id)
	: title(t), author(a), subject(s), book_id(id) {}
};

// 打印书籍信息的函数，接受一个指向 Books 结构体的指针
void printBookInfo(const Books *book) {
    cout << "书籍标题: " << book->title << endl;
    cout << "书籍作者: " << book->author << endl;
    cout << "书籍类目: " << book->subject << endl;
    cout << "书籍 ID: " << book->book_id << endl;
}

int main() {
    // 创建两本书的对象
    Books Book1("C++ 教程", "Runoob", "编程语言", 12345);
    Books Book2("CSS 教程", "Runoob", "前端技术", 12346);

    // 使用指针指向这两本书的对象
    Books *ptrBook1 = &Book1;
    Books *ptrBook2 = &Book2;

    // 输出书籍信息，传递指针
    printBookInfo(ptrBook1);
    printBookInfo(ptrBook2);

    return 0;
}