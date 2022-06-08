import scrapy
from pic16bproject.items import Pic16BprojectItem

class courtscraper(scrapy.Spider):
    name = 'court_spider'
    
    start_urls = ['https://www.supremecourt.gov/opinions/slipopinion/20']

    def parse_start_url(self, response):
        
        return self.parse(response)

    def parse(self, response):
        #months = response.css("div#accordion a")
        cases = response.css("td a")
        pdfs = [a.attrib["href"] for a in cases] 
        prefix = "https://www.supremecourt.gov"
        pdfs_urls = [prefix + suffix for suffix in pdfs]

        yield {
            "links": pdfs_urls
        }

        for url in pdfs_urls:
            item = Pic16BprojectItem() #define it items.py
            item['file_urls'] = [url]
            yield item

    def next_parse(self, response):
        next_page = response.css('div.col-md-12 a::attr(href)').extract() #do i need [0]^M
        yield scrapy.Request(next_page, callback=self.next_parse)

#need to go to next page

#original code for original url
 #   def parse(self, response):

  #       pdfs = [a.attrib["href"] for a in response.css("div#accordion2 a")]
   #      prefix = "https://www.supremecourt.gov/opinions/"
    #     pdfs_urls = [prefix + suffix for suffix in pdfs]


     #    for url in pdfs_urls:
      #      item = Pic16BprojectItem() #define it items.py
       #     item['file_urls'] = [url]
        #    yield item


    # valid_extensions = [".pdf"]

   































 # prelim = response.css("div#accordian2")
        # pdfs = [a.attrib["href"] for a in response.css("div#panel-body a")]

        # for the links to pdfs
        # a id, href
        
        
        #get
        # <li>
        # a id = "#" href = "link"

        #deny_urls = [
        # " https://www.supremecourt.gov/opinions/preliminaryprint/"]

        #valid_extensions = [".pdf"]